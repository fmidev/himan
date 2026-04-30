#include "file_accessor.h"
#include "s3.h"

using namespace himan;

buffer ReadFromLocalFile(const file_information& finfo)
{
	FILE* fp = fopen(finfo.file_location.c_str(), "rb");

	if (!fp)
	{
		throw std::runtime_error("Failed to open file: " + finfo.file_location);
	}

	long length = 0;

	if (finfo.offset && finfo.length)
	{
		long offset = finfo.offset.value();
		length = finfo.length.value();
		fseek(fp, offset, SEEK_SET);
	}
	else
	{
		// read whole file
		fseek(fp, 0, SEEK_END);
		length = ftell(fp);
		if (length < 0)
		{
			fclose(fp);
			throw std::runtime_error("ftell failed for file: " + finfo.file_location);
		}
		rewind(fp);
	}

	buffer buf;
	buf.data = reinterpret_cast<unsigned char*>(malloc(length));

	if (!buf.data)
	{
		fclose(fp);
		throw std::runtime_error("Memory allocation failed for file: " + finfo.file_location);
	}

	buf.length = length;

	fread(buf.data, buf.length, 1, fp);
	fclose(fp);

	return buf;
}

buffer file_accessor::Read(const file_information& finfo) const
{
	switch (finfo.storage_type)
	{
		case kLocalFileSystem:
			return ReadFromLocalFile(finfo);
		case kS3ObjectStorageSystem:
			return s3::ReadFile(finfo);
		default:
			throw std::runtime_error("Unsupported storage system");
	}
}
