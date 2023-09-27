#include "spiller.h"
#include "util.h"
#include <filesystem>
#include <map>
#include <mutex>
#include <atomic>

#ifdef HAVE_CEREAL

static std::atomic<bool> enabled{false};
static std::mutex spillMutex;
static std::map<std::string, std::string> spilledFiles;

using namespace himan;

std::string SpillDirectory(bool create = true)
{
	std::string tmpdir = "/tmp";

	try
	{
		tmpdir = util::GetEnv("HIMAN_TEMP_DIRECTORY");
	}
	catch (...)
	{
	}

	tmpdir = fmt::format("{}/himan-spill-{}", tmpdir, getpid());

	if (create)
	{
		if (std::filesystem::is_directory(tmpdir) == false)
		{
			std::filesystem::create_directory(tmpdir);
		}
	}
	return tmpdir;
}

bool spiller::Enabled()
{
	return enabled; 
}

void spiller::RemoveFromFileName(const std::string& fileName)
{
	std::filesystem::remove(fileName);

	std::lock_guard<std::mutex> lock(spillMutex);

	for (const auto& m : spilledFiles)
	{
		if (m.second == fileName)
		{
			spilledFiles.erase(m.first);
			break;
		}
	}
	if (spilledFiles.size() == 0)
	{
		std::filesystem::remove_all(std::filesystem::path{SpillDirectory(false)}.string());
		enabled = false;
	}
}

void spiller::RemoveFromUniqueName(const std::string& uniqueName)
{
	std::lock_guard<std::mutex> lock(spillMutex);

	for (const auto& m : spilledFiles)
	{
		if (m.first == uniqueName)
		{
			std::filesystem::remove(m.second);
			spilledFiles.erase(uniqueName);
			break;
		}
	}
	if (spilledFiles.size() == 0)
	{
		std::filesystem::remove_all(std::filesystem::path{SpillDirectory(false)}.string());
		enabled = false;
	}
}

void spiller::RemoveAll()
{
	std::lock_guard<std::mutex> lock(spillMutex);

	std::filesystem::remove_all(std::filesystem::path{SpillDirectory(false)}.string());
	spilledFiles.clear();
	enabled = false;
}
template <typename T>
std::shared_ptr<info<T>> spiller::ReadFromFileName(const std::string& fileName)
{
	auto anInfo = std::make_shared<himan::info<T>>();

	std::lock_guard<std::mutex> lock(spillMutex);

	for (const auto& m : spilledFiles)
	{
		if (m.second == fileName)
		{
			std::ifstream infile(fileName, std::ios::binary);
			cereal::BinaryInputArchive iarchive(infile);
			iarchive(anInfo);
			break;
		}
	}

	return anInfo;
}

template std::shared_ptr<info<double>> spiller::ReadFromFileName(const std::string&);
template std::shared_ptr<info<float>> spiller::ReadFromFileName(const std::string&);
template std::shared_ptr<info<short>> spiller::ReadFromFileName(const std::string&);
template std::shared_ptr<info<unsigned char>> spiller::ReadFromFileName(const std::string&);

template <typename T>
std::shared_ptr<info<T>> spiller::ReadFromUniqueName(const std::string& uniqueName)
{
	auto anInfo = std::make_shared<himan::info<T>>();

	std::lock_guard<std::mutex> lock(spillMutex);
	std::string fileName;

	try
	{
		fileName = spilledFiles.at(uniqueName);
	}
	catch (const std::out_of_range& e)
	{
		return nullptr;
	}

	{
		std::ifstream infile(fileName, std::ios::binary);
		cereal::BinaryInputArchive iarchive(infile);
		iarchive(anInfo);
	}

	return anInfo;
}

template std::shared_ptr<info<double>> spiller::ReadFromUniqueName(const std::string&);
template std::shared_ptr<info<float>> spiller::ReadFromUniqueName(const std::string&);
template std::shared_ptr<info<short>> spiller::ReadFromUniqueName(const std::string&);
template std::shared_ptr<info<unsigned char>> spiller::ReadFromUniqueName(const std::string&);

template <typename T>
std::string spiller::Write(std::shared_ptr<info<T>>& anInfo)
{
	const std::string spillFile = fmt::format("{}/{:010d}", SpillDirectory(), rand());
	std::ofstream outfile(spillFile, std::ios::binary);
	cereal::BinaryOutputArchive archive(outfile);

	// extract only currently active info and set file type to double
	auto newInfo =
	    std::make_shared<info<double>>(anInfo->ForecastType(), anInfo->Time(), anInfo->Level(), anInfo->Param());
	newInfo->Producer(anInfo->Producer());

	auto b = std::make_shared<base<double>>();
	b->grid = std::shared_ptr<grid>(anInfo->Grid()->Clone());
	b->data = anInfo->Data();
	newInfo->Base(b);
	archive(newInfo);
	//        itsLogger.Debug(fmt::format("Cache is full, spilling {} to file {}", util::UniqueName(*info), spillFile));

	enabled = true;
	std::lock_guard<std::mutex> lock(spillMutex);
	spilledFiles[util::UniqueName(*anInfo)] = spillFile;
	return spillFile;
}
template std::string spiller::Write(std::shared_ptr<info<double>>&);
template std::string spiller::Write(std::shared_ptr<info<float>>&);
template std::string spiller::Write(std::shared_ptr<info<short>>&);
template std::string spiller::Write(std::shared_ptr<info<unsigned char>>&);

#endif
