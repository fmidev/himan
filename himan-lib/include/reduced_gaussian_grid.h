/**
 * @File:   reduced_gaussian_grid.h
 */

#ifndef REDUCED_GAUSSIAN_GRID_H
#define REDUCED_GAUSSIAN_GRID_H

#include "grid.h"
#include "serialization.h"
#include <map>

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6))
#define override  // override specifier not support until 4.8
#endif

namespace himan
{
class reduced_gaussian_grid : public irregular_grid
{
   public:
	reduced_gaussian_grid();
	virtual ~reduced_gaussian_grid() = default;

	reduced_gaussian_grid(const reduced_gaussian_grid& other);
	reduced_gaussian_grid& operator=(const reduced_gaussian_grid& other) = delete;

	std::string ClassName() const override
	{
		return "himan::gaussian_grid";
	}

	int N() const;
	void N(int theN);

	size_t Size() const override;

	std::vector<int> NumberOfPointsAlongParallels() const;
	void NumberOfPointsAlongParallels(const std::vector<int>& theNumberOfPointsAlongParallels);

	std::vector<size_t> AccumulatedPointsAlongParallels() const;

	std::vector<double> Latitudes() const;

	point FirstPoint() const override;
	point LastPoint() const override;

	std::unique_ptr<grid> Clone() const override;

	std::ostream& Write(std::ostream& file) const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	point LatLon(size_t locationIndex) const override;

	size_t Hash() const override;

	point LatLon(size_t x, size_t y) const;
	size_t LocationIndex(size_t x, size_t y) const;

   private:
	static std::vector<double> GetLatitudes(int theN);
	bool EqualsTo(const reduced_gaussian_grid& other) const;

	std::vector<int> itsNumberOfPointsAlongParallels;
	std::vector<size_t> itsAccumulatedPointsAlongParallels;
	std::vector<double> itsLatitudes;

	int itsN;

	static std::map<int, std::vector<double>> cachedLatitudes;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<irregular_grid>(this), CEREAL_NVP(itsN), CEREAL_NVP(itsNumberOfPointsAlongParallels),
		   CEREAL_NVP(itsN), CEREAL_NVP(itsAccumulatedPointsAlongParallels));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const reduced_gaussian_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::reduced_gaussian_grid);
#endif

#endif /* REDUCED_GAUSSIAN_GRID_H */
