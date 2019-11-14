#ifdef HAVE_S3
#include "s3.h"
#include "debug.h"
#include "util.h"
#include <iostream>
#include <libs3.h>
#include <mutex>
#include <string.h>  // memcpy

using namespace himan;

static std::once_flag oflag;

const char* host = 0;
const char* access_key = 0;
const char* secret_key = 0;
thread_local S3Status statusG = S3StatusOK;

void CheckS3Error(S3Status errarg, const char* file, const int line);

#define S3_CHECK(errarg) CheckS3Error(errarg, __FILE__, __LINE__)

inline void CheckS3Error(S3Status errarg, const char* file, const int line)
{
	if (errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << S3_get_status_name(errarg) << std::endl;
		himan::Abort();
	}
}

S3Status responsePropertiesCallback(const S3ResponseProperties* properties, void* callbackData)
{
	return S3StatusOK;
}

static void responseCompleteCallback(S3Status status, const S3ErrorDetails* error, void* callbackData)
{
	statusG = status;
	return;
}

thread_local S3ResponseHandler responseHandler = {&responsePropertiesCallback, &responseCompleteCallback};

static S3Status getObjectDataCallback(int bufferSize, const char* buffer, void* callbackData)
{
	himan::buffer* ret = static_cast<himan::buffer*>(callbackData);

	ret->data = static_cast<unsigned char*>(realloc(ret->data, ret->length + bufferSize));
	memcpy(ret->data + ret->length, buffer, bufferSize);
	ret->length += bufferSize;

	return S3StatusOK;
}

void Initialize()
{
	call_once(oflag, [&]() {

		access_key = getenv("S3_ACCESS_KEY_ID");
		secret_key = getenv("S3_SECRET_ACCESS_KEY");
		host = getenv("S3_HOSTNAME");

		if (!host)
		{
			host = "s3.us-east-1.amazonaws.com";
		}

		logger logr("s3");

		if (!access_key)
		{
			logr.Fatal("Environment variable S3_ACCESS_KEY_ID not defined");
			himan::Abort();
		}

		if (!secret_key)
		{
			logr.Fatal("Environment variable S3_SECRET_ACCESS_KEY not defined");
			himan::Abort();
		}

		S3_CHECK(S3_initialize("s3", S3_INIT_ALL, host));
	});
}

buffer s3::ReadFile(const file_information& fileInformation)
{
	Initialize();

	S3GetObjectHandler getObjectHandler = {responseHandler, &getObjectDataCallback};

	auto tokens = util::Split(fileInformation.file_location, "/", false);

	const std::string bucket = tokens[0];
	tokens.erase(std::begin(tokens), std::begin(tokens) + 1);

	std::string key;
	for (const auto& piece : tokens)
	{
		if (!key.empty())
			key += "/";
		key += piece;
	}

	buffer ret;

	// clang-format off

	S3BucketContext bucketContext = 
	{
		host,
		bucket.c_str(),
		S3ProtocolHTTP,
		S3UriStylePath,
		access_key,
		secret_key,
		0
	};

	// clang-format on

	logger logr("s3");

	int count = 0;
	do
	{
		if (count > 0)
			sleep(count);
		const unsigned long offset = fileInformation.offset.get();
		const unsigned long length = fileInformation.length.get();

		S3_get_object(&bucketContext, key.c_str(), NULL, offset, length, NULL, &getObjectHandler, &ret);
		logr.Trace("Reading from host=" + std::string(host) + " bucket=" + bucket + " key=" + key + " " +
		           std::to_string(offset) + ":" + std::to_string(length) + " (" + S3_get_status_name(statusG) + ")");
		count++;
	} while (S3_status_is_retryable(statusG) && count < 3);

	switch (statusG)
	{
		case S3StatusOK:
			break;
		case S3StatusInternalError:
			logr.Error(std::string(S3_get_status_name(statusG)) + ": is there a proxy blocking the connection?");
			throw himan::kFileDataNotFound;
		case S3StatusFailedToConnect:
			logr.Error(std::string(S3_get_status_name(statusG)) + ": is proxy required but not set?");
			throw himan::kFileDataNotFound;
		default:
			logr.Error(S3_get_status_name(statusG));
			throw himan::kFileDataNotFound;
	}

	if (ret.length == 0)
	{
		throw himan::kFileDataNotFound;
	}

	return ret;
}
#else
buffer s3::ReadFile(const file_information& fileInformation)
{
	throw std::runtime_error("S3 support not compiled");
}
#endif
