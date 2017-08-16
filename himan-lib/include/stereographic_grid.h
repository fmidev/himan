/*
 * File:   stereographic_grid.h
 *
 */

#ifndef STEREOGRAPHIC_GRID_H
#define STEREOGRAPHIC_GRID_H

#include "grid.h"
#include "logger.h"
#include "serialization.h"
#include <string>

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6))
#define override  // override specifier not support until 4.8
#endif

class NFmiGrid;
namespace himan
{
class stereographic_grid : public grid
{
   public:
	stereographic_grid();
	stereographic_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopRight,
	                   double theOrientation = kHPMissingValue);

	virtual ~stereographic_grid();
	/**
	 * @brief Copy constructor for stereographic_grid
	 *
	 * When stereographic_grid is copied, the contents (ie. class d_matrix_t) is copied as
	 * well.
	 *
	 * @param other
	 */

	stereographic_grid(const stereographic_grid& other);
	stereographic_grid& operator=(const stereographic_grid& other) = delete;

	std::string ClassName() const override { return "himan::stereographic_grid"; }
	std::ostream& Write(std::ostream& file) const override;

	size_t Ni() const override;

	size_t Nj() const override;

	size_t Size() const override;

	double Di() const override;
	double Dj() const override;

	void Ni(size_t theNi);
	void Nj(size_t theNj);

	void Di(double theDi);
	void Dj(double theDj);

	point BottomLeft() const override;
	point TopRight() const override;

	void BottomLeft(const point& theBottomLeft);
	void TopRight(const point& theTopRight);

	void Orientation(double theOrientation);
	double Orientation() const;

	point FirstPoint() const override;
	point LastPoint() const override;

	virtual HPScanningMode ScanningMode() const override;
	virtual void ScanningMode(HPScanningMode theScanningMode) override;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	bool Swap(HPScanningMode newScanningMode) override;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;

	stereographic_grid* Clone() const override;

   private:
	bool SetCoordinates();
	void CreateAreaAndGrid() const;

	bool EqualsTo(const stereographic_grid& other) const;

	mutable std::unique_ptr<NFmiGrid> itsStereGrid;

	point itsBottomLeft;
	point itsTopRight;
	//	point itsBottomRight;
	//	point itsTopLeft;

	double itsOrientation;

	mutable double itsDi;
	mutable double itsDj;

	size_t itsNi;
	size_t itsNj;
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<grid>(this), CEREAL_NVP(itsBottomLeft), CEREAL_NVP(itsTopRight),
		   CEREAL_NVP(itsOrientation), CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const stereographic_grid& ob) { return ob.Write(file); }
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::stereographic_grid);
#endif

#endif /* STEREOGRAPHIC_GRID_H */
