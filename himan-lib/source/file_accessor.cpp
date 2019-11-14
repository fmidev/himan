#include "file_accessor.h"
#include "s3.h"

using namespace himan;

buffer ReadFromLocalFile(const file_information& finfo)
{
	FILE* fp = fopen(finfo.file_location.c_str(), "rb");

	long offset = 0, length = 0;

	if (finfo.offset && finfo.length)
	{
		offset = finfo.offset.get();
		length = finfo.length.get();
		fseek(fp, offset, SEEK_SET);
	}
	else
	{
		// read whole file
		fseek(fp, 0, SEEK_END);
		length = ftell(fp);
		rewind(fp);
	}

	buffer buf;
	buf.data = new unsigned char[length];
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
