#include "s3.h"
using namespace himan;

#ifdef HAVE_S3
#include "debug.h"
#include "timer.h"
#include "util.h"
#include <iostream>
#include <libs3.h>
#include <mutex>
#include <regex>
#include <string.h>  // memcpy

namespace
{
static std::once_flag oflag;

const char* access_key = 0;
const char* secret_key = 0;
const char* security_token = 0;
S3Protocol default_protocol = S3ProtocolHTTPS;
thread_local S3Status statusG = S3StatusOK;

void CheckS3Error(S3Status errarg, const char* file, const int line);

#define S3_CHECK(errarg) CheckS3Error(errarg, __FILE__, __LINE__)

void HandleS3Error(himan::logger& logr, const std::string& host, const std::string& bucket, const std::string& key,
                   const std::string& op)
{
	logr.Error(fmt::format("{} operation for host={} bucket={} object={} failed", op, host, bucket, key));

	const std::string errorstr = S3_get_status_name(statusG);
	switch (statusG)
	{
		case S3StatusInternalError:
			logr.Error(fmt::format("{}: is there a proxy blocking the connection?", errorstr));
			throw himan::kFileDataNotFound;
		case S3StatusFailedToConnect:
			logr.Error(fmt::format("{}: is proxy required but not set?", errorstr));
			throw himan::kFileDataNotFound;
		case S3StatusErrorInvalidAccessKeyId:
			logr.Error(fmt::format(
			    "{}: are Temporary Security Credentials used without security token (env: S3_SESSION_TOKEN)?",
			    errorstr));
			throw himan::kFileDataNotFound;
		case S3StatusErrorAccessDenied:
			logr.Error(fmt::format(
			    "{}: invalid credentials or access protocol (http vs https, hint: set env variable S3_PROTOCOL)?",
			    errorstr));
			throw himan::kFileDataNotFound;
		case S3StatusHttpErrorBadRequest:
			logr.Error(
			    fmt::format("{}: make sure correct session credentials are used (access_key: {}..., secret_key: {}...",
			                errorstr, std::string(access_key).substr(0, 10), std::string(secret_key).substr(0, 10)));
			throw himan::kFileDataNotFound;
		default:
			logr.Error(fmt::format("S3 error: {}", errorstr));
			throw himan::kFileDataNotFound;
	}
}

std::string StripProtocol(const std::string& str)
{
	const static std::regex r("^(https)|(http)|(s3)*://");

	return regex_replace(str, r, "");
}

std::vector<std::string> GetBucketAndFileName(const std::string& fullFileName)
{
	std::vector<std::string> ret;

	auto fileName = StripProtocol(fullFileName);

	// erase forward slash if exists (s3 buckets can't start with /)
	if (fileName[0] == '/')
	{
		fileName = fileName.erase(0, 1);
	}

	auto tokens = util::Split(fileName, "/");

	ret.push_back(tokens[0]);
	tokens.erase(std::begin(tokens), std::begin(tokens) + 1);

	std::string key;
	for (const auto& piece : tokens)
	{
		if (!key.empty())
		{
			key += "/";
		}
		key += piece;
	}

	ret.push_back(key);
	return ret;
}

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

	if (status != S3StatusOK && error->message != nullptr)
	{
		logger logr("s3");
		logr.Error(fmt::format("{} resource: {} further details: {}", error->message,
		                       error->resource == nullptr ? "NULL" : error->resource,
		                       error->furtherDetails == nullptr ? "NULL" : error->furtherDetails));
	}
	return;
}

thread_local S3ResponseHandler responseHandler = {&responsePropertiesCallback, &responseCompleteCallback};

static S3Status getObjectDataCallback(int bufferSize, const char* buffer, void* callbackData)
{
	himan::buffer* ret = static_cast<himan::buffer*>(callbackData);

	// extend callback buffer to accomodate the newest chunk of data
	void* ptr = realloc(ret->data, ret->length + bufferSize);

	if (ptr == nullptr)
	{
		logger logr("s3");
		logr.Error("getObjectDataCallback: Memory allocation failed");
		return S3StatusAbortedByCallback;
	}

	ret->data = static_cast<unsigned char*>(ptr);

	if (bufferSize > 0)
	{
		// copy the new data to the end of the buffer
		memcpy(ret->data + ret->length, buffer, bufferSize);
		ret->length += bufferSize;
	}

	return S3StatusOK;
}

static S3Status headObjectPropertiesCallback(const S3ResponseProperties* properties, void* callbackData)
{
	auto objectSize = static_cast<long unsigned int*>(callbackData);

	if (properties->contentLength)
	{
		*objectSize = properties->contentLength;
	}

	return S3StatusOK;
}

void Initialize()
{
	call_once(
	    oflag,
	    [&]()
	    {
		    access_key = getenv("S3_ACCESS_KEY_ID");
		    secret_key = getenv("S3_SECRET_ACCESS_KEY");
		    security_token = getenv("S3_SESSION_TOKEN");

		    logger logr("s3");

		    if (!access_key || !secret_key)
		    {
			    logr.Info(
			        "S3_ACCESS_KEY_ID and/or S3_SECRET_ACCESS_KEY missing, only unauthenticated access is possible");
		    }

		    try
		    {
			    const auto envproto = himan::util::GetEnv("S3_PROTOCOL");
			    if (envproto == "https")
			    {
				    default_protocol = S3ProtocolHTTPS;
			    }
			    else if (envproto == "http")
			    {
				    default_protocol = S3ProtocolHTTP;
			    }
			    else
			    {
				    logr.Warning(fmt::format("Unrecognized value found from env variable S3_PROTOCOL: '{}'", envproto));
			    }
		    }
		    catch (const std::invalid_argument& e)
		    {
		    }

		    S3_CHECK(S3_initialize("s3", S3_INIT_ALL, NULL));
	    });
}

std::string ReadAWSRegionFromHostname(const std::string& hostname)
{
	if (hostname.find("amazonaws.com") != std::string::npos)
	{
		// extract region name from host name, assuming aws hostname like
		// s3.us-east-1.amazonaws.com

		auto tokens = util::Split(hostname, ".");

		logger logr("s3");

		if (tokens.size() != 4)
		{
			logr.Fatal("Hostname does not follow pattern s3.<regionname>.amazonaws.com");
		}
		else
		{
			logr.Trace(fmt::format("s3 authentication hostname: {}", tokens[1]));
			return tokens[1];
		}
	}

	return std::string();
}

struct write_data
{
	himan::buffer buffer;
	size_t write_ptr;
};

static int putObjectDataCallback(int bufferSize, char* buffer, void* callbackData)
{
	write_data* data = static_cast<write_data*>(callbackData);
	int bytesToWrite = 0;

	if (data->buffer.length)
	{
		const unsigned long buffer_len = data->buffer.length;
		const unsigned int read_buffer_len = static_cast<unsigned int>(bufferSize);
		bytesToWrite = static_cast<int>((buffer_len > read_buffer_len) ? read_buffer_len : buffer_len);
		memcpy(buffer, data->buffer.data + data->write_ptr, bytesToWrite);
		data->write_ptr += bytesToWrite;
		data->buffer.length -= bytesToWrite;
	}

	return bytesToWrite;
}

S3Protocol GetProtocol(const std::string& endpoint)
{
	S3Protocol protocol = default_protocol;

	if (endpoint.find("http://") != std::string::npos)
	{
		protocol = S3ProtocolHTTP;
	}

	return protocol;
}

S3BucketContext GetBucketContext(const std::string& host, const std::string& bucket, const std::string& region)
{
	// clang-format off

#ifdef S3_DEFAULT_REGION

        S3BucketContext bucketContext =
        {
                host.c_str(),
                bucket.c_str(),
                GetProtocol(host),
                S3UriStylePath,
                access_key,
                secret_key,
                security_token,
                // check size, otherwise c_str() will return random garbage from some
                // uninitialized memory location
                region.size() == 0 ? nullptr : region.c_str()
        };
#else

	S3BucketContext bucketContext =
	{
		host.c_str(),
		bucket.c_str(),
		GetProtocol(host),
		S3UriStylePath,
		access_key,
		secret_key,
		security_token
	};
#endif
	// clang-format on

	return bucketContext;
}

}  // namespace

buffer s3::ReadFile(const file_information& fileInformation)
{
	Initialize();
	logger logr("s3");

	S3GetObjectHandler getObjectHandler = {responseHandler, &getObjectDataCallback};

	const auto bucketAndFileName = GetBucketAndFileName(fileInformation.file_location);
	const auto bucket = bucketAndFileName[0];
	const auto key = bucketAndFileName[1];

	if (fileInformation.file_server.empty())
	{
		logr.Error("Empty s3 host name given, don't know where to read data from");
		throw himan::kFileDataNotFound;
	}

	buffer ret;
	auto host = StripProtocol(fileInformation.file_server);
	const auto region = ReadAWSRegionFromHostname(host);

	S3BucketContext bucketContext = GetBucketContext(host, bucket, region);

	const unsigned long length = fileInformation.length.value();

	int count = 0;
	do
	{
		if (count > 0)
		{
			sleep(2 * count);
		}
		const unsigned long offset = fileInformation.offset.value();

#ifdef S3_DEFAULT_REGION
		S3_get_object(&bucketContext, key.c_str(), NULL, offset, length, NULL, 0, &getObjectHandler, &ret);
#else
		S3_get_object(&bucketContext, key.c_str(), NULL, offset, length, NULL, &getObjectHandler, &ret);
#endif
		count++;
	} while (S3_status_is_retryable(statusG) && count < 3);

	switch (statusG)
	{
		case S3StatusOK:
			break;
		default:
			HandleS3Error(logr, fileInformation.file_server, bucket, key, "Read");
			break;
	}

	if (ret.length == 0)
	{
		throw himan::kFileDataNotFound;
	}

	if (length > 0 && ret.length != length)
	{
		logr.Error(fmt::format("Expected {} bytes, got {} bytes", fileInformation.length.value(), ret.length));
		throw himan::kFileDataNotFound;
	}
	return ret;
}

void s3::WriteObject(const std::string& objectName, const buffer& buff)
{
	Initialize();

	const auto bucketAndFileName = GetBucketAndFileName(objectName);
	const auto bucket = bucketAndFileName[0];
	const auto key = bucketAndFileName[1];

	auto host = StripProtocol(util::GetEnv("S3_HOSTNAME"));
	const auto region = ReadAWSRegionFromHostname(host);

	logger logr("s3");

	const auto bucketContext = GetBucketContext(host, bucket, region);

	S3PutObjectHandler putObjectHandler = {responseHandler, &putObjectDataCallback};

	write_data data;
	data.buffer.data = buff.data;
	data.buffer.length = buff.length;
	data.write_ptr = 0;

	timer t(true);

	int count = 0;
	do
	{
		if (count > 0)
		{
			sleep(2 * count);
		}

#ifdef S3_DEFAULT_REGION
		S3_put_object(&bucketContext, key.c_str(), buff.length, NULL, NULL, 0, &putObjectHandler, &data);
#else
		S3_put_object(&bucketContext, key.c_str(), buff.length, NULL, NULL, &putObjectHandler, &data);
#endif

		count++;
	} while (S3_status_is_retryable(statusG) && count < 3);

	// remove pointer to original buff so that double free doesn't occur
	data.buffer.data = 0;

	switch (statusG)
	{
		case S3StatusOK:
		{
			t.Stop();
			const double time = static_cast<double>(t.GetTime());
			const double size = util::round(static_cast<double>(buff.length) / 1024. / 1024., 1);
			logr.Info(fmt::format("Wrote {} MB in {} ms ({:.1f} MBps) to file s3://{}/{}", size, time,
			                      size / (time * 0.001), bucket, key));
		}
		break;
		default:
			HandleS3Error(logr, host, bucket, key, "Write");
			break;
	}
}

bool s3::Exists(const std::string& objectName)
{
	Initialize();

	const auto host = StripProtocol(util::GetEnv("S3_HOSTNAME"));

	const auto bucketAndFileName = GetBucketAndFileName(objectName);
	const auto bucket = bucketAndFileName[0];
	const auto key = bucketAndFileName[1];
	const auto region = ReadAWSRegionFromHostname(host);

	const auto bucketContext = GetBucketContext(host, bucket, region);
	const int timeoutms = 5000;

	S3_head_object(&bucketContext, key.c_str(), NULL, timeoutms, &responseHandler, NULL);

	himan::logger logr("s3");

	switch (statusG)
	{
		case S3StatusOK:
			return true;
		case S3StatusErrorNoSuchKey:
		case S3StatusHttpErrorNotFound:
			return false;
		default:
			HandleS3Error(logr, host, bucket, key, "Head");
			himan::Abort();
	}
}

long unsigned int s3::ObjectSize(const std::string& objectName)
{
	Initialize();

	const auto host = StripProtocol(util::GetEnv("S3_HOSTNAME"));

	const auto bucketAndFileName = GetBucketAndFileName(objectName);
	const auto bucket = bucketAndFileName[0];
	const auto key = bucketAndFileName[1];
	const auto region = ReadAWSRegionFromHostname(host);

	const auto bucketContext = GetBucketContext(host, bucket, region);
	const int timeoutms = 5000;

	long unsigned int size = 0;

	S3ResponseHandler responseHandler = {&headObjectPropertiesCallback, &responseCompleteCallback};

	S3_head_object(&bucketContext, key.c_str(), NULL, timeoutms, &responseHandler, &size);

	himan::logger logr("s3");

	switch (statusG)
	{
		case S3StatusOK:
			return size;
		default:
			HandleS3Error(logr, host, bucket, key, "Size");
			himan::Abort();
	}
}

#else
buffer s3::ReadFile(const file_information& fileInformation)
{
	throw std::runtime_error("S3 support not compiled");
}
void s3::WriteObject(const std::string& objectName, const buffer& buff)
{
	throw std::runtime_error("S3 support not compiled");
}
bool s3::Exists(const std::string& objectName)
{
	throw std::runtime_error("S3 support not compiled");
}
#endif
