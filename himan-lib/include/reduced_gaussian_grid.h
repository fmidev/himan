/**
 * @File:   reduced_gaussian_grid.h
 */

#ifndef REDUCED_GAUSSIAN_GRID_H
#define REDUCED_GAUSSIAN_GRID_H

#include "grid.h"
#include "serialization.h"

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6))
#define override  // override specifier not support until 4.8
#endif

namespace himan
{
class reduced_gaussian_grid : public grid
{
   public:
	reduced_gaussian_grid();
	virtual ~reduced_gaussian_grid() {}
	reduced_gaussian_grid(const reduced_gaussian_grid& other);
	reduced_gaussian_grid& operator=(const reduced_gaussian_grid& other) = delete;

	std::string ClassName() const { return "himan::gaussian_grid"; }
	point BottomLeft() const override;
	point TopRight() const override;
	point TopLeft() const;
	point BottomRight() const;

	void BottomLeft(const point& theBottomLeft);
	void TopRight(const point& theTopRight);
	void BottomRight(const point& theBottomRight);
	void TopLeft(const point& theTopLeft);

	int N() const;
	void N(int theN);

	void Nj(size_t theNj);

	size_t Ni() const override;
	size_t Nj() const override;
	size_t Size() const override;

	double Di() const override;
	double Dj() const override;
	bool Swap(HPScanningMode newScanningMode) override;

	std::vector<int> NumberOfPointsAlongParallels() const;
	void NumberOfPointsAlongParallels(std::vector<int> theNumberOfPointsAlongParallels);

	reduced_gaussian_grid* Clone() const override;

	std::ostream& Write(std::ostream& file) const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const;

	point FirstPoint() const;
	point LastPoint() const;

	/**
	 * @brief Return value of given grid point coordinates
	 */

	double Value(size_t x, size_t y) const;

   private:
	bool EqualsTo(const reduced_gaussian_grid& other) const;
	point LatLon(size_t x, size_t y) const;
	void UpdateCoordinates() const;

	int itsN;
	std::vector<int> itsNumberOfPointsAlongParallels;

	size_t itsNj;

	mutable point itsBottomLeft;
	mutable point itsTopRight;
	mutable point itsBottomRight;
	mutable point itsTopLeft;

	mutable double itsDj;
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<grid>(this), CEREAL_NVP(itsN), CEREAL_NVP(itsNumberOfPointsAlongParallels),
		   CEREAL_NVP(itsNj), CEREAL_NVP(itsBottomLeft), CEREAL_NVP(itsTopRight), CEREAL_NVP(itsBottomRight),
		   CEREAL_NVP(itsTopLeft), CEREAL_NVP(itsDj));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const reduced_gaussian_grid& ob) { return ob.Write(file); }
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::reduced_gaussian_grid);
#endif

#endif /* REDUCED_GAUSSIAN_GRID_H */
