#pragma once

template <typename T>
info<T>::info(const std::vector<forecast_type>& ftypes, const std::vector<forecast_time>& times,
              const std::vector<level>& levels, const std::vector<param>& pars)
    : info<T>()
{
	Set<forecast_type>(ftypes);
	Set<forecast_time>(times);
	Set<level>(levels);
	Set<param>(pars);

	itsDimensions.resize(Size<forecast_type>() * Size<forecast_time>() * Size<level>() * Size<param>());

	First<param>();
	First<forecast_time>();
	First<level>();
	First<forecast_type>();
}

template <typename T>
info<T>::info(const forecast_type& ftype, const forecast_time& time, const level& lvl, const param& par)
    : info(std::vector<forecast_type>({ftype}), std::vector<forecast_time>({time}), std::vector<level>({lvl}),
           std::vector<param>({par}))

{
}

template <typename T>
info<T>::info(const info<T>& other)
    // Iterators are COPIED
    : itsLevelIterator(other.itsLevelIterator),
      itsTimeIterator(other.itsTimeIterator),
      itsParamIterator(other.itsParamIterator),
      itsForecastTypeIterator(other.itsForecastTypeIterator),
      itsDimensions(other.itsDimensions),
      itsProducer(other.itsProducer),
      itsLocationIndex(other.itsLocationIndex)
{
	itsLogger = logger("info");
}

template <typename T>
template <typename V>
info<T>::info(const info<V>& other)
    : itsLevelIterator(other.template Iterator<level>()),
      itsTimeIterator(other.template Iterator<forecast_time>()),
      itsParamIterator(other.template Iterator<param>()),
      itsForecastTypeIterator(other.template Iterator<forecast_type>()),
      itsProducer(other.Producer()),
      itsLocationIndex(other.LocationIndex())
{
	itsLogger = logger("info");

	itsDimensions.resize(other.DimensionSize());

	for (size_t i = 0; i < itsDimensions.size(); i++)
	{
		const auto& ob = other.itsDimensions[i];

		if (ob == nullptr || ob->grid == nullptr)
		{
			continue;
		}

		auto b = std::make_shared<himan::base<T>>();
		b->grid = std::shared_ptr<himan::grid>(ob->grid->Clone());
		b->data = ob->data;

		itsDimensions[i] = b;
	}
}

template <typename T>
inline std::ostream& operator<<(std::ostream& file, const info<T>& ob)
{
	return ob.Write(file);
}

template <typename T>
inline size_t info<T>::Index(size_t forecastTypeIndex, size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
	ASSERT(forecastTypeIndex != kIteratorResetValue);
	ASSERT(timeIndex != kIteratorResetValue);
	ASSERT(levelIndex != kIteratorResetValue);
	ASSERT(paramIndex != kIteratorResetValue);

	return (paramIndex * itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() +
	        levelIndex * itsForecastTypeIterator.Size() * itsTimeIterator.Size() +
	        timeIndex * itsForecastTypeIterator.Size() + forecastTypeIndex);
}

template <typename T>
std::ostream& info<T>::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << itsProducer;

	file << itsParamIterator;
	file << itsLevelIterator;
	file << itsTimeIterator;
	file << itsForecastTypeIterator;

	for (size_t i = 0; i < itsDimensions.size(); i++)
	{
		if (itsDimensions[i]->grid)
		{
			file << *itsDimensions[i]->grid;
		}
		if (itsDimensions[i]->pdata)
		{
			file << "__itsPackedData__ " << itsDimensions[i]->pdata->HasData() << std::endl;
		}
		file << itsDimensions[i]->data;
	}

	return file;
}

template <typename T>
template <typename U>
void info<T>::Regrid(const std::vector<U>& newDim)
{
	size_t ftypesize = Size<forecast_type>();
	size_t timesize = Size<forecast_time>();
	size_t lvlsize = Size<level>();
	size_t parsize = Size<param>();

	if (std::is_same<U, forecast_type>::value)
	{
		ftypesize = newDim.size();
	}
	else if (std::is_same<U, forecast_time>::value)
	{
		timesize = newDim.size();
	}
	else if (std::is_same<U, level>::value)
	{
		lvlsize = newDim.size();
	}
	else if (std::is_same<U, param>::value)
	{
		parsize = newDim.size();
	}

	std::vector<std::shared_ptr<base<T>>> newDimensions(ftypesize * timesize * lvlsize * parsize);

	First<forecast_type>();
	First<forecast_time>();
	First<level>();
	Reset<param>();

	while (Next())
	{
		if (IsValidGrid() == false)
		{
			continue;
		}

		size_t newI = (Index<param>() * ftypesize * timesize * lvlsize + Index<level>() * ftypesize * timesize +
		               Index<forecast_time>() * ftypesize + Index<forecast_type>());

		newDimensions[newI] = std::make_shared<base<T>>(std::shared_ptr<grid>(Grid()->Clone()), Data());
	}

	itsDimensions = move(newDimensions);
	First();  // "Factory setting"
}

template <typename T>
void info<T>::Create(std::unique_ptr<grid> baseGrid, bool createDataBackend)
{
	ASSERT(baseGrid);

	itsDimensions.resize(itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() *
	                     itsParamIterator.Size());

	Reset<forecast_type>();

	while (Next<forecast_type>())
	{
		Reset<forecast_time>();

		while (Next<forecast_time>())
		{
			Reset<level>();

			while (Next<level>())
			{
				Reset<param>();

				while (Next<param>())
				// Create empty placeholders
				{
					auto g = std::shared_ptr<grid>(baseGrid->Clone());
					auto b = std::make_shared<base<T>>(g, matrix<T>());

					Base(b);

					if (baseGrid->Class() == kRegularGrid)
					{
						if (createDataBackend)
						{
							const regular_grid* regGrid(dynamic_cast<const regular_grid*>(baseGrid.get()));
							Data().Resize(regGrid->Ni(), regGrid->Nj());
						}
					}
					else if (baseGrid->Class() == kIrregularGrid)
					{
						Data().Resize(Grid()->Size(), 1, 1);
					}
					else
					{
						itsLogger.Fatal("Invalid grid type");
						Abort();
					}
				}
			}
		}
	}

	First();
}

template <typename T>
void info<T>::Create(std::shared_ptr<base<T>> baseGrid, bool createDataBackend)
{
	ASSERT(baseGrid);

	itsDimensions.resize(itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() *
	                     itsParamIterator.Size());

	Reset<forecast_type>();

	while (Next<forecast_type>())
	{
		Reset<forecast_time>();

		while (Next<forecast_time>())
		{
			Reset<level>();

			while (Next<level>())
			{
				Reset<param>();

				while (Next<param>())
				// Create empty placeholders
				{
					auto g = std::shared_ptr<grid>(baseGrid->grid->Clone());
					auto b = std::make_shared<base<T>>(g, matrix<T>(baseGrid->data));

					Base(b);

					if (baseGrid->grid->Class() == kRegularGrid)
					{
						if (createDataBackend)
						{
							const regular_grid* regGrid(dynamic_cast<const regular_grid*>(baseGrid->grid.get()));
							Data().Resize(regGrid->Ni(), regGrid->Nj());
						}
					}
					else if (baseGrid->grid->Class() == kIrregularGrid)
					{
						Data().Resize(Grid()->Size(), 1, 1);
					}
					else
					{
						itsLogger.Fatal("Invalid grid type");
						Abort();
					}
				}
			}
		}
	}

	First();
}

template <typename T>
void info<T>::Merge(std::shared_ptr<info> otherInfo)
{
	Reset();

	otherInfo->template Reset<forecast_type>();

	// X = forecast type
	// Y = time
	// Z = level
	// Å = param

	while (otherInfo->template Next<forecast_type>())
	{
		if (itsForecastTypeIterator.Add(otherInfo->template Value<forecast_type>()))  // no duplicates
		{
			ReIndex(Size<forecast_type>() - 1, Size<forecast_time>(), Size<level>(), Size<param>());
		}

		if (!Find<forecast_type>(otherInfo->template Value<forecast_type>()))
		{
			itsLogger.Fatal("Unable to set forecast type, merge failed");
			Abort();
		}

		otherInfo->template Reset<forecast_time>();

		while (otherInfo->template Next<forecast_time>())
		{
			if (itsTimeIterator.Add(otherInfo->template Value<forecast_time>()))  // no duplicates
			{
				ReIndex(Size<forecast_type>(), Size<forecast_time>() - 1, Size<level>(), Size<param>());
			}

			if (!Find<forecast_time>(otherInfo->template Value<forecast_time>()))
			{
				itsLogger.Fatal("Unable to set time, merge failed");
				Abort();
			}

			otherInfo->template Reset<level>();

			while (otherInfo->template Next<level>())
			{
				if (itsLevelIterator.Add(otherInfo->template Value<level>()))  // no duplicates
				{
					ReIndex(Size<forecast_type>(), Size<forecast_time>(), Size<level>() - 1, Size<param>());
				}

				if (!Find<level>(otherInfo->template Value<level>()))
				{
					itsLogger.Fatal("Unable to set level, merge failed");
					Abort();
				}

				otherInfo->template Reset<param>();

				while (otherInfo->template Next<param>())
				{
					if (itsParamIterator.Add(otherInfo->template Value<param>()))  // no duplicates
					{
						ReIndex(Size<forecast_type>(), Size<forecast_time>(), Size<level>(), Size<param>() - 1);
					}

					if (!Find<param>(otherInfo->template Value<param>()))
					{
						itsLogger.Fatal("Unable to set param, merge failed");
						Abort();
					}

					Base(std::make_shared<base<T>>(std::shared_ptr<grid>(otherInfo->Grid()->Clone()),
					                               otherInfo->Data()));
				}
			}
		}
	}
}

template <typename T>
void info<T>::Merge(std::vector<std::shared_ptr<info>>& otherInfos)
{
	for (size_t i = 0; i < otherInfos.size(); i++)
	{
		Merge(otherInfos[i]);
	}
}

template <typename T>
void info<T>::First()
{
	First<level>();
	First<param>();
	First<forecast_time>();
	First<forecast_type>();
	FirstLocation();
}

template <typename T>
void info<T>::Reset()
{
	Reset<level>();
	Reset<param>();
	Reset<forecast_time>();
	Reset<forecast_type>();
	ResetLocation();
}

template <typename T>
bool info<T>::NextLocation()
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLocationIndex = 0;  // ResetLocation() has been called before this function
	}

	else
	{
		itsLocationIndex++;
	}

	size_t locationSize = Data().Size();

	if (itsLocationIndex >= locationSize)
	{
		itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;

		return false;
	}

	return true;
}

template <typename T>
void info<T>::ReIndex(size_t oldForecastTypeSize, size_t oldTimeSize, size_t oldLevelSize, size_t oldParamSize)
{
	std::vector<std::shared_ptr<base<T>>> theDimensions(Size<forecast_type>() * Size<forecast_time>() * Size<level>() *
	                                                    Size<param>());

	for (size_t a = 0; a < oldForecastTypeSize; a++)
	{
		for (size_t b = 0; b < oldTimeSize; b++)
		{
			for (size_t c = 0; c < oldLevelSize; c++)
			{
				for (size_t d = 0; d < oldParamSize; d++)
				{
					size_t index = d * oldForecastTypeSize * oldTimeSize * oldLevelSize +
					               c * oldForecastTypeSize * oldTimeSize + b * oldForecastTypeSize + a;

					size_t newIndex = Index(a, b, c, d);
					theDimensions[newIndex] = itsDimensions[index];
				}
			}
		}
	}

	itsDimensions = theDimensions;
}

template <typename T>
point info<T>::LatLon() const
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLogger.Error("Location iterator position is not set");
		return point();
	}

	return Grid()->LatLon(itsLocationIndex);
}

template <typename T>
station info<T>::Station() const
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLogger.Error("Location iterator position is not set");
		return station();
	}
	else if (Grid()->Class() != kIrregularGrid)
	{
		itsLogger.Error("regular_grid does not hold station information");
		return station();
	}

	return std::dynamic_pointer_cast<himan::point_list>(Grid())->Station(itsLocationIndex);
}

template <typename T>
void info<T>::Clear()
{
	itsDimensions.clear();

	itsParamIterator.Clear();
	itsLevelIterator.Clear();
	itsTimeIterator.Clear();
	itsForecastTypeIterator.Clear();
}

template <typename T>
bool info<T>::Next()
{
	// Innermost

	if (Next<param>())
	{
		return true;
	}

	// No more params at this forecast type/level/time combination; rewind param iterator

	First<param>();

	if (Next<level>())
	{
		return true;
	}

	// No more levels at this forecast type/time combination; rewind level iterator

	First<level>();

	if (Next<forecast_time>())
	{
		return true;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	First<forecast_time>();

	if (Next<forecast_type>())
	{
		return true;
	}

	return false;
}

template <typename T>
void info<T>::FirstValidGrid()
{
	for (Reset<param>(); Next<param>();)
	{
		if (IsValidGrid())
		{
			return;
		}
	}

	itsLogger.Fatal("A dimension with no valid infos? Madness!");
	Abort();
}
