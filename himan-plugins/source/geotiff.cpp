#include "geotiff.h"
#include "cpl_conv.h"  // for CPLMalloc()
#include "file_accessor.h"
#include "gdal_frmts.h"
#include "grid.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "logger.h"
#include "plugin_factory.h"
#include "producer.h"
#include "reduced_gaussian_grid.h"
#include "s3.h"
#include "stereographic_grid.h"
#include "timer.h"
#include "transverse_mercator_grid.h"
#include "util.h"
#include <algorithm>
#include <filesystem>
#include <ogr_spatialref.h>
#include <regex>
#include <thread>

#include "plugin_factory.h"
#include "radon.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "gdal_priv.h"

#pragma GCC diagnostic pop

using namespace himan;
using namespace himan::plugin;

namespace
{
// from s3.cpp
std::string StripProtocol(const std::string& str)
{
	const static std::regex r("^(https)|(http)|(s3)*://");

	return regex_replace(str, r, "");
}
}  // namespace

struct GDALDatasetCloser
{
	void operator()(GDALDataset* ds) const
	{
		GDALClose(ds);
	}
};

typedef std::unique_ptr<GDALDataset, GDALDatasetCloser> GDALDatasetPtr;

void CheckGDALError(OGRErr errarg, const char* file, const int line);

#define GDAL_CHECK(errarg) CheckGDALError(errarg, __FILE__, __LINE__)

inline void CheckGDALError(OGRErr errarg, const char* file, const int line)
{
	if (errarg != OGRERR_NONE)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << CPLGetLastErrorMsg() << std::endl;
		himan::Abort();
	}
}

template <typename T>
GDALDataType TypeToGDALType();

template <>
GDALDataType TypeToGDALType<unsigned char>()
{
	return GDT_Byte;
}

template <>
GDALDataType TypeToGDALType<short>()
{
	return GDT_Int16;
}

template <>
GDALDataType TypeToGDALType<float>()
{
	return GDT_Float32;
}

template <>
GDALDataType TypeToGDALType<double>()
{
	return GDT_Float64;
}

template <typename T>
T ConvertTo(const std::string& str)
{
	std::istringstream ss(str);
	T num;
	ss >> num;
	return num;
}

static std::once_flag oflag;

void CreateDirectory(const std::string& filename)
{
	namespace fs = std::filesystem;
	fs::path pathname(filename);

	if (!pathname.parent_path().empty() && !fs::is_directory(pathname.parent_path()))
	{
		fs::create_directories(pathname.parent_path());
	}
}

geotiff::geotiff()
{
	call_once(oflag,
	          [&]()
	          {
		          GDALRegister_GTiff();
		          GDALRegister_COG();

		          // Check environment for AWS variables
		          // GDAL requires           Himan uses
		          // * AWS_S3_ENDPOINT       S3_HOSTNAME
		          // * AWS_ACCESS_KEY_ID     S3_ACCESS_KEY_ID
		          // * AWS_SECRET_ACCESS_KEY S3_SECRET_ACCESS_KEY
		          // * AWS_SESSION_TOKEN     S3_SESSION_TOKEN
		          //
		          // if latter is found, copy it to former
		          //
		          // also remove any protocol (https:// or s3://) from S3_HOSTNAME

		          const std::vector<std::pair<std::string, std::string>> keys{
		              {"S3_HOSTNAME", "AWS_S3_ENDPOINT"},
		              {"S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"},
		              {"S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"},
		              {"S3_SESSION_TOKEN", "AWS_SESSION_TOKEN"}};

		          for (const auto& key : keys)
		          {
			          try
			          {
				          const std::string val = StripProtocol(util::GetEnv(key.first));
				          setenv(key.second.c_str(), val.c_str(), 0);
			          }
			          catch (...)
			          {
			          }
		          }
	          });

	itsLogger = logger("geotiff");
}

std::pair<HPWriteStatus, file_information> geotiff::ToFile(info<double>& anInfo)
{
	return ToFile<double>(anInfo);
}

void WriteAreaAndGrid(GDALDataset& ds, const himan::regular_grid& g, const producer& prod, const configuration& conf)
{
	const point fp = g.Projected(g.TopLeft());

	double adfGeoTransform[6] = {fp.X(), g.Di(), 0, fp.Y(), 0, -1 * g.Dj()};

	GDAL_CHECK(ds.SetGeoTransform(adfGeoTransform));

	OGRSpatialReference sp;

	GDAL_CHECK(sp.importFromProj4(g.Proj4String().c_str()));

	// If earth shape is WGS84, set datum also to WGS84, since AFAIK
	// no other datum uses it as ellipsoid.

	if (g.EarthShape().Name() == "WGS84")
	{
		sp.SetWellKnownGeogCS("WGS84");
	}

	const std::string geom = conf.TargetGeomName().empty() ? prod.Name() : conf.TargetGeomName();
	GDAL_CHECK(sp.SetProjCS(geom.c_str()));
	GDAL_CHECK(ds.SetSpatialRef(&sp));
}

template <typename T>
void WriteData(GDALDataset& ds, const info<T>& anInfo, int bandNo)
{
	const int ni = static_cast<int>(anInfo.Data().SizeX());
	const int nj = static_cast<int>(anInfo.Data().SizeY());
	const GDALDataType dtype = TypeToGDALType<T>();

	GDALRasterBand* poBand = ds.GetRasterBand(bandNo);

	if (bandNo == 1)
	{
		// Only for first band, because otherwise:
		// band 2: Setting nodata to nan on band 2, but band 1 has nodata at nan. The TIFFTAG_GDAL_NODATA only support
		// one value per dataset. This value of nan will be used for all bands on re-opening
		GDAL_CHECK(poBand->SetNoDataValue(anInfo.Data().MissingValue()));
	}

	matrix<T> values = anInfo.Data();

	if (dynamic_cast<regular_grid*>(anInfo.Grid().get())->ScanningMode() != kTopLeft)
	{
		util::Flip<T>(values);
	}

	if (poBand->RasterIO(GF_Write, 0, 0, ni, nj, values.ValuesAsPOD(), ni, nj, dtype, 0, 0) != OGRERR_NONE)
	{
		logger logr("geotiff");
		logr.Error("File write failed");
		himan::Abort();
	}
}

void WriteBandMetadata(GDALRasterBand* b, const forecast_type& ftype, const forecast_time& ftime, const level& lvl,
                       const param& par)
{
	GDAL_CHECK(b->SetMetadataItem("forecast_type", static_cast<std::string>(ftype).c_str(), nullptr));
	GDAL_CHECK(
	    b->SetMetadataItem("origin_time", ftime.OriginDateTime().String("%Y-%m-%dT%H:%M:%S+00:00").c_str(), nullptr));
	GDAL_CHECK(
	    b->SetMetadataItem("valid_time", ftime.ValidDateTime().String("%Y-%m-%dT%H:%M:%S+00:00").c_str(), nullptr));
	GDAL_CHECK(b->SetMetadataItem("step", ftime.Step().String("%02h:%02M:%02S").c_str(), nullptr));
	GDAL_CHECK(b->SetMetadataItem("level", static_cast<std::string>(lvl).c_str(), nullptr));
	GDAL_CHECK(b->SetMetadataItem("param_name", par.Name().c_str(), nullptr));
	if (par.Aggregation().Type() != kUnknownAggregationType)
	{
		GDAL_CHECK(b->SetMetadataItem("aggregation", static_cast<std::string>(par.Aggregation()).c_str(), nullptr));
	}
	if (par.ProcessingType().Type() != kUnknownProcessingType)
	{
		GDAL_CHECK(
		    b->SetMetadataItem("processing_type", static_cast<std::string>(par.ProcessingType()).c_str(), nullptr));
	}
}

template <typename T>
std::pair<HPWriteStatus, file_information> geotiff::ToFile(info<T>& anInfo)
{
	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger.Error(fmt::format("Unable to write irregular grid of type {} to geotiff",
		                            HPGridTypeToString.at(anInfo.Grid()->Type())));
		throw kInvalidWriteOptions;
	}

	if (itsWriteOptions.configuration->WriteStorageType() == kS3ObjectStorageSystem ||
	    itsWriteOptions.configuration->WriteMode() != kSingleGridToAFile)
	{
		return std::make_pair(HPWriteStatus::kPending, file_information());
	}

	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

	file_information finfo;
	finfo.file_location = util::MakeFileName(anInfo, *itsWriteOptions.configuration);
	finfo.file_type = kGeoTIFF;
	finfo.storage_type = itsWriteOptions.configuration->WriteStorageType();

	// We use Create() method as there is nothing to copy from
	// (CreateCopy() being the alternative)

	const regular_grid* g = dynamic_cast<regular_grid*>(anInfo.Grid().get());

	// Enable compression
	char** opts = NULL;
	opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
	const GDALDataType dtype = TypeToGDALType<T>();

	CreateDirectory(finfo.file_location);
	auto ds = GDALDatasetPtr(driver->Create(finfo.file_location.c_str(), static_cast<int>(anInfo.Data().SizeX()),
	                                        static_cast<int>(anInfo.Data().SizeY()), 1, dtype, opts));

	WriteAreaAndGrid(*ds, *g, anInfo.Producer(), *itsWriteOptions.configuration);
	GDAL_CHECK(ds->SetMetadataItem("producer_id", fmt::format("{}", anInfo.Producer().Id()).c_str(), nullptr));
	WriteBandMetadata(ds->GetRasterBand(1), anInfo.ForecastType(), anInfo.Time(), anInfo.Level(), anInfo.Param());
	WriteData(*ds, anInfo, 1);

	itsLogger.Info(fmt::format("Wrote file '{}'", finfo.file_location));

	return std::make_pair(HPWriteStatus::kFinished, finfo);
}

template std::pair<HPWriteStatus, file_information> geotiff::ToFile<double>(info<double>&);
template std::pair<HPWriteStatus, file_information> geotiff::ToFile<float>(info<float>&);
template std::pair<HPWriteStatus, file_information> geotiff::ToFile<short>(info<short>&);
template std::pair<HPWriteStatus, file_information> geotiff::ToFile<unsigned char>(info<unsigned char>&);

template <typename T>
std::vector<std::pair<HPWriteStatus, file_information>> geotiff::ToFile(const std::vector<info<T>>& infos)
{
	// No "pending" checking here

	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

	std::map<std::string, std::vector<size_t>> list;

	for (size_t i = 0; i < infos.size(); i++)
	{
		const auto& info = infos[i];
		const std::string fname = util::MakeFileName<T>(info, *itsWriteOptions.configuration);
		auto& elem = list[fname];
		elem.push_back(i);
	}

	std::vector<std::pair<HPWriteStatus, file_information>> ret(infos.size());

	// If writing to s3, first write all data to a memory location
	// where it can be picked up. Use gdal's /vsimem feature for this.

	const std::string vrt =
	    (itsWriteOptions.configuration->WriteStorageType() == kS3ObjectStorageSystem) ? "/vsimem/" : "";

	for (const auto& m : list)
	{
		const auto& first = infos[m.second[0]];

		file_information finfo;
		finfo.file_location = m.first;
		finfo.file_type = kGeoTIFF;
		finfo.storage_type = itsWriteOptions.configuration->WriteStorageType();

		const regular_grid* g = dynamic_cast<regular_grid*>(first.Grid().get());

		// Enable compression
		char** opts = NULL;
		opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
		const GDALDataType dtype = TypeToGDALType<T>();

		if (itsWriteOptions.configuration->WriteStorageType() != kS3ObjectStorageSystem)
		{
			CreateDirectory(finfo.file_location);
		}

		auto ds = GDALDatasetPtr(driver->Create(
		    fmt::format("{}{}", vrt, finfo.file_location).c_str(), static_cast<int>(first.Data().SizeX()),
		    static_cast<int>(first.Data().SizeY()), static_cast<int>(m.second.size()), dtype, opts));
		if (!ds)
		{
			himan::Abort();
		}

		GDAL_CHECK(ds->SetMetadataItem("producer_id", fmt::format("{}", first.Producer().Id()).c_str(), nullptr));
		WriteAreaAndGrid(*ds, *g, first.Producer(), *itsWriteOptions.configuration);

		int j = 1;
		for (const size_t i : m.second)
		{
			WriteBandMetadata(ds->GetRasterBand(j), infos[i].ForecastType(), infos[i].Time(), infos[i].Level(),
			                  infos[i].Param());
			WriteData(*ds, infos[i], j);
			finfo.message_no = j++;
			ret[i] = std::make_pair(HPWriteStatus::kFinished, finfo);
		}

		if (itsWriteOptions.configuration->WriteStorageType() == kS3ObjectStorageSystem)
		{
			ds->FlushCache();

			// Earlier data was written to memory, now get a pointer to that memory block
			// and pass it to himan::s3

			VSILFILE* inmem = VSIFOpenL(fmt::format("{}/{}", vrt, finfo.file_location).c_str(), "rb");

			himan::buffer buff;

			// Get file size
			VSIFSeekL(inmem, 0, SEEK_END);
			buff.length = VSIFTellL(inmem);
			VSIFSeekL(inmem, 0, SEEK_SET);

			itsLogger.Trace(fmt::format("In-mem file size is {} bytes", buff.length));

			buff.data = static_cast<unsigned char*>(malloc(buff.length));

			// Read contents
			VSIFReadL(buff.data, buff.length, 1, inmem);

			s3::WriteObject(finfo.file_location, buff);
			itsLogger.Info(fmt::format("Wrote file 's3://{}'", finfo.file_location));
		}
		else
		{
			itsLogger.Info(fmt::format("Wrote file '{}'", finfo.file_location));
		}
	}

	return ret;
}

template std::vector<std::pair<HPWriteStatus, file_information>> geotiff::ToFile<double>(
    const std::vector<info<double>>&);
template std::vector<std::pair<HPWriteStatus, file_information>> geotiff::ToFile<float>(
    const std::vector<info<float>>&);
template std::vector<std::pair<HPWriteStatus, file_information>> geotiff::ToFile<short>(
    const std::vector<info<short>>&);
template std::vector<std::pair<HPWriteStatus, file_information>> geotiff::ToFile<unsigned char>(
    const std::vector<info<unsigned char>>&);

std::unique_ptr<grid> ReadAreaAndGrid(GDALDataset* ds)
{
	logger log("geotiff");
	const int ni = ds->GetRasterXSize(), nj = ds->GetRasterYSize();
	double adfGeoTransform[6];
	if (ds->GetGeoTransform(adfGeoTransform) != CE_None)
	{
		log.Error("File does not contain geo transformation coefficients");
		throw kFileMetaDataNotFound;
	}

	const double di = adfGeoTransform[1];
	const double dj = fabs(adfGeoTransform[5]);
	const point fp(adfGeoTransform[0], adfGeoTransform[3]);
	const HPScanningMode sm = (adfGeoTransform[5] < 0) ? kTopLeft : kBottomLeft;
	ASSERT(di > 0);

	std::string proj = ds->GetProjectionRef();

	if (proj.empty())
	{
		log.Fatal("File does not contain spatial metadata");
		throw kFileMetaDataNotFound;
	}

	OGRSpatialReference spRef(proj.c_str());
	const char* projptr = spRef.GetAttrValue("PROJECTION");

	if (projptr != nullptr)
	{
		const std::string projection = spRef.GetAttrValue("PROJECTION");

		if (projection == SRS_PT_LAMBERT_AZIMUTHAL_EQUAL_AREA)
		{
			return std::unique_ptr<lambert_equal_area_grid>(new lambert_equal_area_grid(
			    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(spRef.Clone()), true));
		}
		else if (projection == SRS_PT_TRANSVERSE_MERCATOR)
		{
			return std::unique_ptr<transverse_mercator_grid>(new transverse_mercator_grid(
			    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(spRef.Clone()), true));
		}
		else if (projection == SRS_PT_LAMBERT_CONFORMAL_CONIC_1SP || projection == SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP)
		{
			return std::unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid(
			    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(spRef.Clone()), true));
		}

		log.Error(fmt::format("Unsupported projection: {}", projection));
	}
	else if (spRef.IsGeographic())
	{
		// No projection -- latlon with some datum
		// spRef.GetAttrValue("DATUM|SPHEROID|AUTHORITY")

		OGRErr erra = 0, errb = 0;
		const double A = spRef.GetSemiMajor(&erra);
		const double B = spRef.GetSemiMinor(&errb);

		earth_shape<double> es;

		if (erra != OGRERR_NONE || errb != OGRERR_NONE)
		{
			log.Error("Unable to extract datum information from file");
		}
		else
		{
			es = earth_shape<double>(A, B);
		}

		return std::unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(sm, fp, ni, nj, di, dj, es));
	}
	throw kFileMetaDataNotFound;
}

param ReadParam(const std::map<std::string, std::string>& meta, const producer& prod, const param& par)
{
	logger logr("geotiff");

	std::string param_value;

	for (const auto& m : meta)
	{
		if (m.first == "param_name")
		{
			param_value = m.second;
			break;
		}
	}

	if (param_value.empty())
	{
		return par;
	}

	auto r = GET_PLUGIN(radon);
	auto parameter = r->RadonDB().GetParameterFromGeoTIFF(prod.Id(), param_value);

	if (parameter.empty() || parameter["name"].empty())
	{
		logr.Warning(
		    fmt::format("Parameter information matching '{}' not found from table 'param_geotiff'", param_value));
		return par;
	}

	param p(parameter["name"]);
	p.Id(std::stoi(parameter["id"]));
	p.InterpolationMethod(par.InterpolationMethod());

	return p;
}

level ReadLevel(const std::map<std::string, std::string>& meta, const level& lvl)
{
	HPLevelType type = HPLevelType::kUnknownLevel;
	double value = kHPMissingValue, value2 = kHPMissingValue;
	for (const auto& m : meta)
	{
		if (m.first == "level")
		{
			const auto tokens = util::Split(m.second, "/");
			type = HPStringToLevelType.at(tokens[0]);
			value = stod(tokens[1]);
			if (tokens.size() == 3)
			{
				value2 = stod(tokens[2]);
			}
			break;
		}
	}

	if (type != kUnknownLevel)
	{
		return level(type, value, value2);
	}

	return lvl;
}

forecast_type ReadForecastType(const std::map<std::string, std::string>& meta, const forecast_type& ftype)
{
	HPForecastType type = HPForecastType::kUnknownType;
	double value = kHPMissingValue;
	for (const auto& m : meta)
	{
		if (m.first == "forecast_type")
		{
			const auto tokens = util::Split(m.second, "/");
			type = HPStringToForecastType.at(tokens[0]);

			if (tokens.size() == 2)
			{
				value = stod(tokens[1]);
			}
			break;
		}
	}

	if (type != kUnknownType)
	{
		return forecast_type(type, value);
	}

	return ftype;
}

void SQLTimeMaskToCTimeMask(std::string& sqlTimeMask)
{
	boost::replace_all(sqlTimeMask, "YYYY", "%Y");
	boost::replace_all(sqlTimeMask, "MM", "%m");
	boost::replace_all(sqlTimeMask, "DD", "%d");
	boost::replace_all(sqlTimeMask, "HH24", "%H");
	boost::replace_all(sqlTimeMask, "MI", "%M");
	boost::replace_all(sqlTimeMask, "SS", "%S");
}

forecast_time ReadTime(const std::map<std::string, std::string>& meta, const forecast_time& ftime)
{
	raw_time origintime = ftime.OriginDateTime();
	raw_time validtime = ftime.ValidDateTime();

	std::string origintimestr, validtimestr, mask;

	for (const auto& m : meta)
	{
		if (m.first == "analysis_time")
		{
			origintimestr = m.second;
		}
		else if (m.first == "valid_time")
		{
			validtimestr = m.second;
		}
		else if (m.first == "time_mask")
		{
			mask = m.second;
		}
	}

	if (boost::to_lower_copy(mask) == "yyyymmddhhmm")
	{
		mask = "%Y%m%d%H%M";
	}
	else if (mask.find("YYYY") != std::string::npos)
	{
		SQLTimeMaskToCTimeMask(mask);
	}

	if (!origintimestr.empty() && !mask.empty())
	{
		origintime = raw_time(origintimestr, mask);
	}

	if (!validtimestr.empty() && !mask.empty())
	{
		validtime = raw_time(validtimestr, mask);
	}

	return forecast_time(origintime, validtime);
}

std::map<std::string, std::string> ParseMetadata(char** mdata, const producer& prod)
{
	std::map<std::string, std::string> ret;

	if (mdata == nullptr)
	{
		return ret;
	}

	// First check keys with Himan-known 'standard' names
	const std::vector<std::string> standardNames{"forecast_type", "level", "param_name", "origin_time", "valid_time"};

	for (const auto& keyName : standardNames)
	{
		const char* m = CSLFetchNameValue(mdata, keyName.c_str());
		if (m != nullptr)
		{
			ret[keyName] = std::string(m);
		}
	}

	std::string query =
	    fmt::format("SELECT attribute, key, mask FROM geotiff_metadata WHERE producer_id = {}", prod.Id());

	auto r = GET_PLUGIN(radon);

	r->RadonDB().Query(query);

	logger log("geotiff");

	while (true)
	{
		const auto row = r->RadonDB().FetchRow();
		if (row.empty())
		{
			break;
		}

		const auto attribute = row[0];
		const auto keyName = row[1];
		const auto keyMask = row[2];

		std::string metadata;
		if (keyName.empty() == false)
		{
			// metadata consists of key=value pairs
			const char* m = CSLFetchNameValue(mdata, keyName.c_str());

			if (m == nullptr)
			{
				log.Trace("Did not find expected key '" + keyName + "' from metadata");
				continue;
			}

			metadata = std::string(m);
		}
		else if (mdata != nullptr)
		{
			// no defined keys for metadata elements
			// in database this means that 'key' column is empty string
			metadata = std::string(*mdata);
		}
		else
		{
			continue;
		}

		// Try to extract information from free-form text fields
		// attribute: a metadata attribute name that Himan understands, like 'analysis_time'
		// keyName: name of the metadata element in geotiff file
		// keyMask: optional mask for the value of the key, if only specific value needs
		//          to be extraced, regular expressions are used

		if (keyMask.empty())
		{
			ret[attribute] = metadata;
		}
		else
		{
			const std::regex re(keyMask);
			std::smatch what;
			if (std::regex_search(metadata, what, re) == false || what.size() == 0)
			{
				log.Warning(fmt::format("Regex did not match for attribute {}", attribute));
				log.Warning(fmt::format("Regex: '{}' Metadata: '{}'", keyMask, metadata));
			}

			if (what.size() != 2)
			{
				log.Fatal(fmt::format("Regex matched too many times: {}", what.size() - 1));
				himan::Abort();
			}

			log.Debug(fmt::format("Regex match for {}: {}", attribute, std::string(what[1])));
			ret[attribute] = what[1];
		}
	}

	return ret;
}

template <typename T>
void ReadData(GDALRasterBand* poBand, matrix<T>& mat, const std::map<std::string, std::string>& meta)
{
	if (meta.find("missing_value") != meta.end())
	{
		mat.MissingValue(ConvertTo<T>(meta.at("missing_value")));
	}
	else
	{
		int ret = 0;
		const T miss = static_cast<T>(poBand->GetNoDataValue(&ret));

		if (ret == 1)
		{
			mat.MissingValue(miss);
		}
	}

	ASSERT(poBand->GetXSize() == static_cast<int>(mat.SizeX()));
	ASSERT(poBand->GetYSize() == static_cast<int>(mat.SizeY()));

	int nXSize = poBand->GetXSize();
	int nYSize = poBand->GetYSize();

	if (poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, mat.ValuesAsPOD(), nXSize, nYSize, TypeToGDALType<T>(), 0, 0) !=
	    CE_None)
	{
		throw std::runtime_error("Read failed");
	}

	const T offset = static_cast<T>(poBand->GetOffset(nullptr));
	const T scale = static_cast<T>(poBand->GetScale(nullptr));

	// Change missingvalue to our own
	mat.MissingValue(MissingValue<T>());

	// Apply scale and base
	if (offset != 0 || scale != 1)
	{
		auto& data = mat.Values();
		for_each(data.begin(), data.end(), [=](T& val) { val = static_cast<T>(val * scale + offset); });
	}
}

std::vector<std::shared_ptr<info<double>>> geotiff::FromFile(const file_information& theInputFile,
                                                             const search_options& options, bool readData) const
{
	return FromFile<double>(theInputFile, options, readData);
}

template <typename T>
std::vector<std::shared_ptr<info<T>>> geotiff::FromFile(const file_information& theInputFile,
                                                        const search_options& options, bool readData) const
{
	const bool validate = options.configuration->ValidateMetadata();

	std::vector<std::shared_ptr<himan::info<T>>> infos;

	auto ParseFileName = [](const file_information& finfo)
	{
		std::string ret = finfo.file_location;

		if (finfo.storage_type == kS3ObjectStorageSystem)
		{
			ret = fmt::format("/vsis3_streaming/{}", StripProtocol(ret));
		}
		return ret;
	};

	auto ds =
	    GDALDatasetPtr(reinterpret_cast<GDALDataset*>(GDALOpen(ParseFileName(theInputFile).c_str(), GA_ReadOnly)));

	if (ds == nullptr)
	{
		itsLogger.Error("Failed to open dataset from " + theInputFile.file_location);
		return infos;
	}

	auto meta = ParseMetadata(ds->GetMetadata(), options.prod);  // Get full dataset metadata

	if (meta.size() == 0)
	{
		itsLogger.Trace(
		    fmt::format("No elements recognized from global metadata from '{}'", theInputFile.file_location));
	}

	auto area = ReadAreaAndGrid(ds.get());

	if (area == nullptr)
	{
		return infos;
	}

	// "first guess" metadata from file metadata

	auto par = ReadParam(meta, options.prod, options.param);
	auto lvl = ReadLevel(meta, options.level);
	auto ftype = ReadForecastType(meta, options.ftype);
	auto ftime = ReadTime(meta, options.time);

	auto MakeInfoFromGeoTIFFBand = [&](GDALRasterBand* poBand) -> std::shared_ptr<info<T>>
	{
		// Read possible metadata from band
		auto bmeta = ParseMetadata(poBand->GetMetadata(), options.prod);

		auto bpar = ReadParam(bmeta, options.prod, par);
		auto blvl = ReadLevel(bmeta, lvl);
		auto bftype = ReadForecastType(bmeta, ftype);
		auto bftime = ReadTime(bmeta, ftime);

		if (bpar == param() || blvl == level() || bftype == forecast_type() || bftime == forecast_time())
		{
			itsLogger.Warning("Failed to gather all required metadata");
			itsLogger.Warning("Param: " + bpar.Name());
			itsLogger.Warning("Level: " + static_cast<std::string>(blvl));
			itsLogger.Warning("Time: " + bftime.OriginDateTime().String() + " step: " + bftime.Step().String("%H:%M"));
			itsLogger.Warning("Forecast type: " + static_cast<std::string>(bftype));
			throw kFileDataNotFound;
		}

		if (validate && options.time != bftime)
		{
			itsLogger.Warning("Time does not match: " + options.time.OriginDateTime().String() + " step " +
			                  options.time.Step().String("%02H:%02M:%02S") + " vs " + bftime.OriginDateTime().String() +
			                  " step " + bftime.Step().String("%02H:%02M:%02S"));
		}
		if (validate && options.level != blvl)
		{
			itsLogger.Warning("Level does not match");
		}
		if (validate && options.ftype != bftype)
		{
			itsLogger.Warning("Forecast type does not match");
		}
		if (validate && options.param != bpar)
		{
			itsLogger.Warning("param does not match: " + options.param.Name() + " vs " + bpar.Name());
		}

		auto anInfo = std::make_shared<info<T>>(bftype, bftime, blvl, bpar);
		auto b = std::make_shared<base<T>>();
		b->grid = std::shared_ptr<grid>(area->Clone());

		anInfo->Create(b, true);
		anInfo->Producer(options.prod);

		if (readData)
		{
			ReadData<T>(poBand, anInfo->Data(), meta);
		}
		return anInfo;
	};

	if (theInputFile.message_no == std::nullopt)
	{
		for (int bandNo = 1; bandNo <= ds->GetRasterCount(); bandNo++)
		{
			itsLogger.Info("Read from file '" + theInputFile.file_location + "' band# " + std::to_string(bandNo));
			GDALRasterBand* poBand = ds->GetRasterBand(bandNo);
			try
			{
				infos.push_back(MakeInfoFromGeoTIFFBand(poBand));
			}
			catch (const HPExceptionType& e)
			{
			}
		}
	}
	else
	{
		itsLogger.Info("Read from file '" + theInputFile.file_location + "' band# " +
		               std::to_string(theInputFile.message_no.value()));
		GDALRasterBand* poBand = ds->GetRasterBand(static_cast<int>(theInputFile.message_no.value()));
		try
		{
			infos.push_back(MakeInfoFromGeoTIFFBand(poBand));
		}
		catch (...)
		{
		}
	}

	return infos;
}

template std::vector<std::shared_ptr<info<double>>> geotiff::FromFile<double>(const file_information&,
                                                                              const search_options&, bool) const;
template std::vector<std::shared_ptr<info<float>>> geotiff::FromFile<float>(const file_information&,
                                                                            const search_options&, bool) const;
template std::vector<std::shared_ptr<info<short>>> geotiff::FromFile<short>(const file_information&,
                                                                            const search_options&, bool) const;
template std::vector<std::shared_ptr<info<unsigned char>>> geotiff::FromFile<unsigned char>(const file_information&,
                                                                                            const search_options&,
                                                                                            bool) const;
