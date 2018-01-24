# Summary

hitool-plugin enables the examination of the state of the atmosphere. It exposes several functions that scan the atmosphere vertically and aggregate the values in some fashion. 

For execution hitool needs hybrid levels: the parameter value and its height. The vertical extent can be limited using either metric or isobaric height. The height range can be set separately per gridpoint or one value for all. Internally hitool always works with whole grids.

Hitool has bindings for Himan lua functionality (luatool).

# Methods

The functions exposed by hitool are listed below. Return value of all functions is a std::vector<double>. When multiple params are given (i.e. vector<param>), hitool will return the first one that is found.

## Maximum value

Return maximum value of some parameter in the given height range.

    VerticalMaximum(param, double lowerHeight, double upperHeight)
    VerticalMaximum(vector<param>, double lowerHeight, double upperHeight)
    VerticalMaximum(param, vector<double> lowerHeight, vector<double> upperHeight)
    VerticalMaximum(vector<param>, vector<double> lowerHeight, vector<double> upperHeight)

Example: Find maximum temperature between 0 and 200 meters.

    // Get and initialize hitool, etc
    auto max = h->VerticalMaximum(param("T-K"), 0, 200);


## Minimum value

Return minimum value of some parameter in the given height range.

    VerticalMinimum(param, double lowerHeight, double upperHeight)
    VerticalMinimum(vector<param>, double lowerHeight, double upperHeight)
    VerticalMinimum(param, vector<double> lowerHeight, vector<double> upperHeight)
    VerticalMinimum(vector<param>, vector<double> lowerHeight, vector<double> upperHeight)

Example: Find minimum total cloudiness between 0 and 500 meters.

    // Get and initialize hitool, etc
    // Not sure if parameter is N-0TO1 or N-PRCNT
    
    vector<param> paramList({param("N-0TO1"), param("N-PRCNT")}); 
    auto min = h->VerticalMinimum(paramList, 0, 500); 

## Mean value

Return mean value of some parameter in the given height range. Mean value calculated using the First mean value theorem for definite integrals so that it is weighted with distance, as generally the distance between hybrid levels is gradually increasing towards the top to atmosphere.

    VerticalAverage(param, double lowerHeight, double upperHeight)
    VerticalAverage(vector<param>, double lowerHeight, double upperHeight)
    VerticalAverage(param, vector<double> lowerHeight, vector<double> upperHeight)
    VerticalAverage(vector<param>, vector<double> lowerHeight, vector<double> upperHeight)

Example: Find mean temperature between 850 and 700 isobars.

    // Get and initialize hitool, etc
    
    h->HeightUnit(kHPa);   
    auto mean = h->VerticalAverage(paramList, 850, 700);

## Sum

Return sum of values of some parameter in the given height range.

    VerticalSum(param, double lowerHeight, double upperHeight)
    VerticalSum(vector<param>, double lowerHeight, double upperHeight)
    VerticalSum(param, vector<double> lowerHeight, vector<double> upperHeight)
    VerticalSum(vector<param>, vector<double> lowerHeight, vector<double> upperHeight)

## Count

Return the number of some specific parameter value found in the given height range.

    VerticalCount(param, double lowerHeight, double upperHeight, double searchValue)
    VerticalCount(vector<param>, double lowerHeight, double upperHeight, double searchValue)
    VerticalCount(param, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue)
    VerticalCount(vector<param>, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue)

Example: Find number of 0 degree isotherms.

    // Get and initialize hitool, etc
    
    auto cnt = h->VerticalCount(param("T-K"), 50, 5000, 273.15);

## Value

Return the value of some parameter at given height. Hitool will automatically narrow the search to the correct height range if necessary information about hybrid level heights is found from database. 

    VerticalValue(param, double searchValue)
    VerticalValue(vector<param>, double searchValue)
    VerticalValue(param, vector<double> searchValue)
    VerticalValue(vector<param>, vector<double> searchValue)

Example: Find the wind speed at boundary layer height.

    // Get and initialize hitool, etc
    
    auto BLH = Fetch(...); // fetch boundary layer height as a vector (individual value for all grid points) 
    auto value = h->VerticalValue(param("FF-MS"), BLH);


## Height

Return the height where some parameters' value crosses the user given threshold. By default the last occurence will be returned, but that can adjusted with argument Nth.

    VerticalHeight(vector<param>, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeight(vector<param>, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)
    VerticalHeight(vector<param>, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeight(param, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeight(param, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeight(param, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)

Example: Find the height of first and second 0 degree isotherms.

    // Get and initialize hitool, etc
    
    auto first = h->VerticalCount(param("T-K"), 50, 5000, 273.15, 1);
    auto second = h->VerticalCount(param("T-K"), 50, 5000, 273.15, 2);

## Height greater than

Return the height where some parameter value was higher than given limit value. This function will also consider the case where the value was found very close to the starting height, or where the value is consistently higher than limit value. For example when searching for the height where total cloudiness > 80%, in the case where a stratus cloud is found the regular height-search does not find the value (because the the cloudiness value did not cross 80% threshold). Argument Nth can have values 0 (= search for the last height where parameter value was encountered) or 1 (= search for the last height where parameter value was encountered).

    VerticalHeightGreaterThan(vector<param>, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightGreaterThan(vector<param>, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightGreaterThan(vector<param>, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeightGreaterThan(param, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeightGreaterThan(param, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightGreaterThan(param, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)

Example: Find the first height where total cloudiness >= 80%, limit search to lowest 100 meters.

    // Get and initialize hitool, etc
    
    auto height = h->VerticalHeightGreaterThan(param("N-PRCNT"), 0, 100, 80, 1);

## Height less than

Return the height where some parameter value was lower than given value. Argument Nth can have values 0 (= search for the last height where parameter value was encountered) or 1 (= search for the last height where parameter value was encountered).

    VerticalHeightLessThan(vector<param>, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightLessThan(vector<param>, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightLessThan(vector<param>, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeightLessThan(param, double lowerHeight, double upperHeight, double searchValue, int Nth)
    VerticalHeightLessThan(param, double lowerHeight, double upperHeight, vector<double> searchValue, int Nth)
    VerticalHeightLessThan(param, vector<double> lowerHeight, vector<double> upperHeight, vector<double> searchValue, int Nth)

Example: Find the last height where wind speed <= 5 m/s , limit search to lowest 100 meters.

    // Get and initialize hitool, etc
    
    auto height = h->VerticalHeightLessThan(param("FF-MS"), 0, 100, 5, 0);

# Per-plugin configuration options

None.
