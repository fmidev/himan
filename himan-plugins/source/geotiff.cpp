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
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <ogr_spatialref.h>

#include "plugin_factory.h"
#include "radon.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "gdal_priv.h"

#pragma GCC diagnostic pop

using namespace himan;
using namespace himan::plugin;

geotiff::geotiff()
{
	itsLogger = logger("geotiff");
}

himan::file_information geotiff::ToFile(info<double>& anInfo)
{
	return ToFile<double>(anInfo);
}

template <typename T>
file_information geotiff::ToFile(info<T>& anInfo)
{
	if (anInfo.Grid()->Class() == kIrregularGrid && anInfo.Grid()->Type() != kReducedGaussian)
	{
		itsLogger.Error("Unable to write irregular grid of type " + HPGridTypeToString.at(anInfo.Grid()->Type()) +
		                " to geotiff");
		throw kInvalidWriteOptions;
	}

	itsLogger.Fatal("No support for writing geotiff data");
	himan::Abort();
}

template file_information geotiff::ToFile<double>(info<double>&);
template file_information geotiff::ToFile<float>(info<float>&);

std::unique_ptr<grid> ReadAreaAndGrid(GDALDataset* poDataset)
{
	logger log("geotiff");
	const int ni = poDataset->GetRasterXSize(), nj = poDataset->GetRasterYSize();
	double adfGeoTransform[6];
	if (poDataset->GetGeoTransform(adfGeoTransform) != CE_None)
	{
		log.Error("File does not contain geo transformation coefficients");
		throw kFileMetaDataNotFound;
	}

	const double di = adfGeoTransform[1];
	const double dj = fabs(adfGeoTransform[5]);
	const point fp(adfGeoTransform[0], adfGeoTransform[3]);
	const HPScanningMode sm = (adfGeoTransform[5] < 0) ? kTopLeft : kBottomLeft;
	ASSERT(di > 0);

	std::string proj = poDataset->GetProjectionRef();

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

		log.Error("Unsupported projection: " + projection);
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
		return par;
	}

	param p(parameter["name"]);
	p.Id(std::stoi(parameter["id"]));
	return p;
}

level ReadLevel(const std::map<std::string, std::string>& meta, const level& lvl)
{
	// Don't know how to deal with this yet
	return lvl;
}

forecast_type ReadForecastType(const std::map<std::string, std::string>& meta, const forecast_type& ftype)
{
	// Don't know how to deal with this yet
	return ftype;
}

void SQLTimeMaskToCTimeMask(std::string& sqlTimeMask)
{
	boost::replace_all(sqlTimeMask, "YYYY", "%Y");
	boost::replace_all(sqlTimeMask, "MM", "%m");
	boost::replace_all(sqlTimeMask, "DD", "%d");
	boost::replace_all(sqlTimeMask, "hh", "%H");
	boost::replace_all(sqlTimeMask, "mm", "%M");
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

	if (mask.find("YYYY") != std::string::npos)
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

template <typename T>
GDALDataType TypeToGDALType();

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

std::map<std::string, std::string> ParseMetadata(char** mdata, const producer& prod)
{
	std::map<std::string, std::string> ret;

	if (mdata == nullptr)
	{
		return ret;
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
			const boost::regex re(keyMask);
			boost::smatch what;
			if (boost::regex_search(metadata, what, re) == false || what.size() == 0)
			{
				log.Warning("Regex did not match for attribute " + attribute);
				log.Warning("Regex: '" + keyMask + "' Metadata: '" + metadata + "'");
			}

			if (what.size() != 2)
			{
				log.Fatal("Regex matched too many times: " + std::to_string(what.size() - 1));
				himan::Abort();
			}

			log.Debug("Regex match for " + attribute + ": " + what[1]);
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

	ASSERT(poBand->GetXSize() == mat.SizeX());
	ASSERT(poBand->GetYSize() == mat.SizeY());

	int nXSize = poBand->GetXSize();
	int nYSize = poBand->GetYSize();

	if (poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, mat.ValuesAsPOD(), nXSize, nYSize, TypeToGDALType<T>(), 0, 0) !=
	    CE_None)
	{
		throw std::runtime_error("Read failed");
	}
	// Change missingvalue to our own
	mat.MissingValue(MissingValue<T>());
}

std::vector<std::shared_ptr<info<double>>> geotiff::FromFile(const file_information& theInputFile,
                                                             const search_options& options, bool validate,
                                                             bool readData) const
{
	return FromFile<double>(theInputFile, options, validate, readData);
}

template <typename T>
std::vector<std::shared_ptr<info<T>>> geotiff::FromFile(const file_information& theInputFile,
                                                        const search_options& options, bool validate,
                                                        bool readData) const
{
	std::vector<std::shared_ptr<himan::info<T>>> infos;

	GDALRegister_GTiff();
	// GDALRegister_COG(); // not working, maybe due to using an oldish gdal version

	std::unique_ptr<GDALDataset, std::function<void(GDALDataset*)>> poDataset(
	    reinterpret_cast<GDALDataset*>(GDALOpen(theInputFile.file_location.c_str(), GA_ReadOnly)),
	    [](GDALDataset* dset) { GDALClose(dset); });

	if (poDataset == nullptr)
	{
		itsLogger.Error("Failed to open dataset from " + theInputFile.file_location);
		return infos;
	}

	auto meta = ParseMetadata(poDataset->GetMetadata(), options.prod);  // Get full dataset metadata

	if (meta.size() == 0)
	{
		itsLogger.Error("Failed to parse metadata from " + theInputFile.file_location);
		return infos;
	}

	auto area = ReadAreaAndGrid(poDataset.get());

	if (area == nullptr)
	{
		return infos;
	}

	// "first guess" metadata from file metadata

	auto par = ReadParam(meta, options.prod, param());
	auto lvl = ReadLevel(meta, options.level);
	auto ftype = ReadForecastType(meta, options.ftype);
	auto ftime = ReadTime(meta, options.time);

	auto MakeInfoFromGeoTIFFBand = [&](GDALRasterBand* poBand) -> std::shared_ptr<info<T>> {

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

	if (theInputFile.message_no == boost::none)
	{
		for (int bandNo = 1; bandNo <= poDataset->GetRasterCount(); bandNo++)
		{
			itsLogger.Info("Read from file '" + theInputFile.file_location + "' band# " + std::to_string(bandNo));
			GDALRasterBand* poBand = poDataset->GetRasterBand(bandNo);
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
		               std::to_string(theInputFile.message_no.get()));
		GDALRasterBand* poBand = poDataset->GetRasterBand(static_cast<int>(theInputFile.message_no.get()));
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
                                                                              const search_options&, bool, bool) const;
template std::vector<std::shared_ptr<info<float>>> geotiff::FromFile<float>(const file_information&,
                                                                            const search_options&, bool, bool) const;
