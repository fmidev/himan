/**
 * @file visibility
 *
 *
 * @date Mar 10, 2016
 * @author aalto
 */

#include "visibility.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "util.h"
#include "level.h"
#include "forecast_time.h"
#include <boost/lexical_cast.hpp>
#include <cmath>

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "fetcher.h"
#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

const string itsName("visibility");

const double defaultVis = 50000;
const double pseudoRR = 0.13;
const double threshold = 1.5;

// Required source parameters

const himan::param PFParam("PRECFORM-N");
const himan::param RHParam("RH-PRCNT");
const himan::param CFParam("CL-FT");
const himan::param TParam("T-K");
const himan::param FFParam("FF-MS");
const himan::param BLHParam("MIXHGT-M");
const himan::param RRParam(himan::param("RRR-KGM2"));
const himan::params NParam({himan::param("N-PRCNT"), himan::param("N-0TO1")});

// ..and their levels
const himan::level NLevel(himan::kHeight, 0, "HEIGHT");
const himan::level RHLevel(himan::kHeight, 2, "HEIGHT");
const himan::level FFLevel(himan::kHeight, 10, "HEIGHT");

visibility::visibility()
{
	itsClearTextFormula = "<algorithm>";
	itsLogger = logger_factory::Instance()->GetLog(itsName);

}

void visibility::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("VV2-M", 407)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void visibility::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));
	
	info_t CFInfo = Fetch(forecastTime, NLevel, CFParam, forecastType, false);
	info_t RHInfo = Fetch(forecastTime, RHLevel, RHParam, forecastType, false);
	info_t PFInfo = Fetch(forecastTime, NLevel, PFParam, forecastType, false);
	info_t FFInfo = Fetch(forecastTime, FFLevel, FFParam, forecastType, false);
	info_t BLHInfo = Fetch(forecastTime, NLevel, BLHParam, forecastType, false);
	info_t TInfo = Fetch(forecastTime, RHLevel, TParam, forecastType, false);
	info_t RRInfo = Fetch(forecastTime, NLevel, RRParam, forecastType, false);

	if (!TInfo || !CFInfo || !RRInfo || !RHInfo || !FFInfo || !PFInfo || !BLHInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}
	
        auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(forecastTime);

	vector<double> stratus;
	vector<double> stratus30;
	vector<double> stratus300;
	vector<double> lowclouds;
	vector<double> highclouds;
	VertMax(myTargetInfo, stratus, NParam, 0, 304);
	VertMax(myTargetInfo, stratus30, NParam, 0, 30);
	VertMax(myTargetInfo, stratus300, NParam, 31, 300);
	VertMax(myTargetInfo, lowclouds, NParam, 60, 600);
	VertMax(myTargetInfo, highclouds, NParam, 6001, 12000);
	
	vector<double> hum25;
	vector<double> hum50;
	vector<double> hum75;
	vector<double> hum100;
	vector<double> hum125;
	VertMax(myTargetInfo, hum25, RHParam, 0, 25);
	VertMax(myTargetInfo, hum50, RHParam, 26, 50);
	VertMax(myTargetInfo, hum75, RHParam, 51, 75);
	VertMax(myTargetInfo, hum100, RHParam, 76, 100);
	VertMax(myTargetInfo, hum125, RHParam, 101, 125);
	
	vector<double> temp25;
	vector<double> temp50;
	vector<double> temp75;
	vector<double> temp100;
	vector<double> temp125;
	VertTMin(myTargetInfo, temp25, 0, 25);
	VertTMin(myTargetInfo, temp50, 26, 50);
	VertTMin(myTargetInfo, temp75, 51, 75);
	VertTMin(myTargetInfo, temp100, 76, 100);
	VertTMin(myTargetInfo, temp125, 101, 125);
	
	vector<double> ffblh;
	VertFFValue(myTargetInfo, ffblh, VEC(BLHInfo));
	
	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo,TInfo,CFInfo,PFInfo,RHInfo,FFInfo,RRInfo)
	{

		size_t i = myTargetInfo->LocationIndex();
		double vis = defaultVis;

		double T = TInfo->Value();
		double CF = CFInfo->Value();
		double PF = PFInfo->Value();
		double RR = RRInfo->Value();
		double FF = FFInfo->Value();
		double RH = RHInfo->Value();
		double strat = stratus[i];
		double strat30 = stratus30[i];
		double strat300 = stratus300[i];
		double lowC = lowclouds[i];
		double highC = highclouds[i];
		double HUM25 = hum25[i];
		double HUM50 = hum50[i];
		double HUM75 = hum75[i];
		double HUM100 = hum100[i];
		double HUM125 = hum125[i];
		double T25 = temp25[i];
		double T50 = temp50[i];
		double T75 = temp75[i];
		double T100 = temp100[i];
		double T125 = temp125[i];
		double FFBLH = ffblh[i];		
		
		if (IsMissingValue({CF, T, FF, RH, RR, strat, HUM25, HUM50, HUM75, HUM100, HUM125, T25, FFBLH}))
		{
			myTargetInfo->Value(vis);
			continue;
		}
		
		assert(T > 200);
		assert(T25 > 200);
		assert(T50 > 200);
		assert(T75 > 200);
		assert(T100 > 200);
		assert(T125 > 200);

		assert(strat <= 1.0);
		assert(strat30 <= 1.0);
		assert(strat300 <= 1.0);
		
		assert(lowC <= 1.0);
		assert(lowC <= 1.0);
		
		assert(HUM25 < 102.);
		assert(HUM50 < 102.);
		assert(HUM75 < 102.);
		assert(HUM100 < 102.);
		assert(HUM125 < 102.);
		
		assert(RR < 50);
		
		T -= himan::constants::kKelvin;
		T25 -= himan::constants::kKelvin;
		T50 -= himan::constants::kKelvin;
		T75 -= himan::constants::kKelvin;
		T100 -= himan::constants::kKelvin;
		T125 -= himan::constants::kKelvin;
		strat *= 100;
		strat30 *= 100;
		strat300 *= 100;
		lowC *= 100;
		highC *= 100;
		
		double RHpre = 85/RH;
		double stNpre = 1;
		double visPre = 1;
		double stHpre = 1;
		
		if (strat >= 50)
		{  
			stNpre = log(50) / log(strat);		
		} 
		
		if (CF <= 500)
		{  
			stHpre = pow((CF / 500), 0.15); 
		}  
		
		// Nakyvyys sateessa
		if (RR > 0)
		{		
		
  			// Drizzle (tai j��t�v� tihku)
  			if (PF == 0 || PF == 4)
  			{
    			// Nakyvyys intensiteetin perusteella
    				visPre = 1/RR*500;
     
    				// Mahdollinen lisahuononnus utuisuuden perusteella
    				visPre = visPre*RHpre*stNpre*stHpre;
  			}
 
  			// Vesisade (tai jäätävä vesisade)
  			if (PF == 1 || PF == 5)
  			{
    				// Nakyvyys intensiteetin perusteella
    				// (kaava antaa ehkä turhan huonoja <4000m näkyvyyksiä reippaassa RR>4 vesisateessa)
    				visPre = 1 / RR*6000+2000;
 
    				// Voimakkaalla sateella jätetään matalimpien stratuksien utuisuusvaikutus huomioimatta
    				// (malleissa usein reippaassa rintamasateessa liian matalaa ~100-200ft stratusta)
    				// Tätä pitää varmasti parantaa/säätää... esim. intensiteetin raja-arvoa. Käyttö myös lumi/räntäsateessa?
    				// IF ((PAR49_HIR>2) AND (par500_HIR<300))
    				//    {  stHpre = 1 }
 
    				// Mahdollinen lisähuononnus utuisuuden perusteella
    				visPre = visPre*RHpre*stNpre*stHpre;
  			}
 
 		 	// Snow
 		 	if (PF == 3)
 		 	{
 		   		// Näkyvyys intensiteetin perusteella
  		  		visPre = 1 / (RR+pseudoRR)*1400;
     
  		  		// Mahdollinen lisähuononnus utuisuuden perusteella
  		  		visPre = visPre*RHpre*stNpre*stHpre;
 		 	}
  
		  	// Sleet
			if (PF == 2)
			{
 		   		// Näkyvyys intensiteetin perusteella
 		   		visPre = 1/(RR+pseudoRR)*2000;
 
 		   		// Mahdollinen lisähuononnus utuisuuden perusteella
 		   		visPre = visPre*RHpre*stNpre*stHpre;
 			}
		
		}
		
		if (visPre > defaultVis )
		{  
			visPre = defaultVis;
		}
		
		double visMist = defaultVis;
	        double stHmist = 1;	
		// SUMUP��TTELY
		
		// Utuisuuskertoimien laskenta stratuksen maaran ja korkeuden perusteella
		// Kertoimia saatamalla voi saataa kunkin parametrin vaikutusta/painoarvoa.
 
		// Näkyvyyden utuisuuskerroin udussa/sumussa sumupilven maaran perusteella [7,0...0,7, kun stN = 0...100%]
		double stNmist = sqrt(50/(strat+1));
 
		// Nakyvyyden utuisuuskerroin udussa/sumussa sumupilvikorkeuden perusteella [ 0,47...1, kun par500 = 50...999ft]
		if (CF < 1000)
		{  
			stHmist = pow((CF/999), 0.25);
		}
 
		// Ehto pilvipaatteyyn menemiselle. Tarkoituksena poistaa mallin virheelliset vain alimman mallipinnan sumupilvitulkinnat.
		// sfcCloud parametri saa arvon 1 jos mallin pilvi on vain alimman 30m korkeudessa. Jos pilvea on yli 1/10 31-300m korkeudessa, sfcCloud saa arvon 0
		double sfcCloud = 0;
		
		if (strat30 > 55 && strat300 < 10)
		{  
			sfcCloud = 1;
		}
 
		// Nakyvyys udussa/sumussa, lasketaan myos heikossa sateessa (tarkoituksena tasoittaa suuria nakyvyysgradientteja sateen reunalla)
		// (ehka syyta rajata vain tilanteisiin, jossa sateen perusteella saatu nakyvyys oli viela >8000?)
 
		if (RR < 0.5 && sfcCloud < 1)
		{	
			double RHmistLim = 90; // Oletus RH:n kynnysarvo utuisuudelle, kun T>=0C [%]
  			// Pakkasella asetetaan utuisuuden kynnysarvo pienemmaksi [%]
  			// Alkuarvauksena yksinkertainen lineaarinen riippuvuus -> kynnysarvon minimi=80%, kun T=-8C
  			if (T < 0)
  			{  
				RHmistLim = 90 + T*1.25;
			}
  			if (RHmistLim < 80)
  			{  
				RHmistLim=80;
			}
 
  			// Nakyvyys udussa/sumussa
 
  			// Yksinkertaistetty kaava, eli lasketaan samalla tavalla riippumatta siita, onko pakkasta vai ei
  			// (pakkaset eri tavalla voisi olla parempi tapa)
  			if (RH > RHmistLim)
  			{   		
      				// Alkuarvo suht. kosteuden perusteella [700-31400m, kun 100<=RH<80]
      				visMist = pow((101 - RH), 1.25) *700;
      				// Lisamuokkaus sumupilven maaran ja korkeuden perusteella
      				visMist = visMist * stNmist * stHmist;
  			}
		}
		
		// Jos sumupilven perusteella laskettu näkyvyys edelleen yli 8km tutkitaan sateilysumujen mahdollisuutta.
		if (visMist > 8000)
		{
		
  		// Säteilysumuun vaikuttavat parametrit ja niiden painoarvot
  		//----------------------------------------------------------------------------------------
  		// pintatuulen nopeus (FF)
  		// ala + keskipilvien maara (par273, par274)
  		// Suhteellisen kosteuden ja lampotilaan kokeellisesti suhteutetun suhteellisen kosteuden erotus on positiivinen
  		// (koska kovemmissa pakkasissa riittavan pienempi suhteellisen kosteuden arvo sumun syntymiselle)
  		// Suhteellinen kosteus alin 125m (suhteutettuna T:en)
  		// Tuulen nopeus rajakerroksen ylaosassa (kuvaa mekaanisen turbulenssin avulla tapahtuvaa kuivemman ilman sekoittumista rajakerroksen yläpuolelta)
   
  		// Naita parametreja painotetaan siten etta arvolla 0 ne eivat esta yhtaan sateilysumun syntymistä ja arvolla 1 ne estavat yksinaan säteilysumun syntymisen.
  		// kokonaisvaikutus sumun syntymiseen saadaan kaikkien ainesosasten yhteisvaikutuksena, siten etta suhteellinen kosteus pinnalla saa kaksinkertaisen painoarvon.
   			// ********* Ala- ja keskipilven määrän vaikutus ---------------------------------------------------
  			//painokerroin ala- ja keskipilvisyydelle 0 = 0, 30 = 0,16, 40 = 0,40, 55 = 1
  			//eli pilvisyys estaa yksinaan sateilysumun synnyn jos sita on yli puoli taivasta (55%).
       		
        
    			double Cloud_coeff = pow((lowC), 3) * 0.000006;
  			 //jos yla- tai alapilvia yksinaan yli 8/10 taivasta -> painokerroin on 1
      			if (highC > 80) 
			{
      				Cloud_coeff = 1;
			}
      			if (lowC > 80) 
			{
      				Cloud_coeff = 1;
			}
         
       			//huolehditaan etta Cloud_coeff arvo pysyy nollan ja ykkosen valilla
        		if (Cloud_coeff < 0)
        		{  
				Cloud_coeff = 0;
			}
        		if (Cloud_coeff > 1)
         		{ 
				Cloud_coeff = 1;
			}
			
			// ********* Pintatuulen nopeuden vaikutus ------------------------------------------------------------
  			// painokerroin tuulen nopeudelle 1 m/s = 0, 1,6m/s = 0.3, 2,6m/s = 0.7  4m/s = 1 (eksponentiaalinen riippuvuus)
  			// eli tuulen nopeus estää säteilysumun synnyn jos se on suurempi kuin 4 m/s
   
        		double Wind_coeff = log(FF*FF)*0.361;
         
        		//huolehditaan etta Wind_coeff arvo pysyy nollan ja ykkosen valilla
        		if (Wind_coeff < 0)
        		{  
				Wind_coeff = 0;
			}
        		if (Wind_coeff > 1)
        		{
				Wind_coeff = 1;
			}
			
			 // ********* Pinnan suhteellisen kosteuden vaikutus -------------------------------------------------------------------
  			// kokeellinen lämpötilaan suhteutettu RH:n alin arvo jossa säteilysumuja voi vielä syntyä (sovitettu excelissä)
  			// tällä suhteelisen kosteuden arvolla painokerroin saa arvon 1 ja jos suhteellinen kosteus on tietyssä lämpötilassa alle tämän arvon säteilysumua ei voi esiintyä.
        
        		double humidity_min =  -0.0028*T*T + 0.3*T + 91.5;
         
        		// kokeellinen lämpötilaan suhteutettu RH:n ylin arvo jossa säteilysumujen esiintyminen on jo todennäköistä (sovitettu excelissä)
        		// tällä suhteelisen kosteuden arvolla painokerroin saa siis arvon 0
        		double humidity_max =  -0.0016*T*T + 0.2*T + 96;
         
        		// Suhteellisen kosteuden minimiarvo
        		if (humidity_min < 75)
        		{  
				humidity_min = 75;
			}
         
        		//painokerroin suhteelliselle kosteudelle
        		double Humidity_coeff = 1 - (RH - humidity_min) / (humidity_max - humidity_min);
         
        		if (Humidity_coeff <0)
        		{  
				Humidity_coeff = 0;
			}
			
			
   			double RH_upper = HUM25*0.1 + HUM50*0.2 +HUM75*0.2 +HUM100*0.2 +HUM125*0.3;
			double T_upper = T25*0.1 +T50*0.2 +T75*0.2 +T100*0.2 +T125*0.3;
             
        		double humidity_min_upper =  -0.0028*T_upper*T_upper + 0.35*T_upper + 92.5;
        		double humidity_max_upper =  -0.0004*T_upper*T_upper + 0.14*T_upper + 97;
          
        		double Humidity_upper_coeff = 1 - (RH_upper - humidity_min_upper) / (humidity_max_upper - humidity_min_upper);
           
        		if (Humidity_upper_coeff < 0)
        		{  
				Humidity_upper_coeff = 0;
			} 
			
			// *************** Rajakerroksen yläosien tuulen nopeuden vaikutus
  			// painokerroin saa arvon 0, kun rajakerroksen yläosan tuuli on 7.5 m/s tai alle. Painokerroin 1 tulee arvolla 12.5 m/s tai yli
  			// Rajakerroksen korkeus PAR180
			
       			double Wind_upper_coeff = log(FFBLH * FFBLH * FFBLH) * 0.67 - 3.91;
      
       			if (Wind_upper_coeff < 0)
       			{  
				Wind_upper_coeff = 0;
			}
  
  			// Näkyvyyden ainesosien yhteenlasketut painokertoimet kuvaavat huonon näkyvyyden mahdollisuutta
  
  			double visibility_sum = Humidity_coeff + Humidity_upper_coeff + Cloud_coeff + Wind_coeff + Wind_upper_coeff;
 
  			// Eri ainesosasten yhteenlasketun summan raja-arvo
  			// threshold = 1.5;
   
  			//huonoja näkyvyyksiä jos "todennäköisyys" alle määritetyn alarajan ja yksittäiset parametrit alle 1
  			// Näkyvyys lasketaan mallin suhteellisen kosteuden ja aikaisemmin lasketun humidity_min parametrin erotuksesta. 
			// humidity_min on lämpötilaan suhteutettu alin suhteellinen kosteus jossa säteilysumuja voi vielä esiintyä.
  			
			if (visibility_sum < threshold && Cloud_coeff < 1 && Wind_coeff < 1 && Humidity_coeff < 1 && Humidity_upper_coeff < 1 && Wind_upper_coeff < 1)
  			{  
				visMist = (8000 - (RH - humidity_min) * 900);
			}
 
  			// Lisähuononnus näkyvyyteen muiden parametrien kautta (tuuli, pilvisyys, auringonsäteily, suhteellinen kosteus alailmakehässä)
  			// parametri saa maksimissaan arvon 1.8 = 80% huononnus näkyvyyteen
  			double extra = 1.8 - (Humidity_upper_coeff + Cloud_coeff + Wind_coeff + Wind_upper_coeff + 0.1) / 3;
  			if (extra < 3 && extra > 0.01)
  			{  
				visMist = visMist/extra;
			}			 
			
		}
	
		// Säteilysumun perusteella laskettu VisMist voi mennä negatiiviseksi ainakin kovilla pakkasilla vuoristossa joten annetaan minimiarvoksi 150m joka vastaa hyvin pienimpiä päättelysssä havaittuja näkyvyyksiä
  		if (visMist < 150)
  		{  
			visMist = 150;
		}
				
		// Lopuksi valitaan sade- ja sumu/utunakyvyyksista huonompi

		if (visMist < visPre)
		{  
		   vis = visMist; 
		}
		else
		{  
		   vis = visPre; 
		} 

		myTargetInfo->Value(vis);
		
	}	
	

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

void visibility::VertMax(shared_ptr<info> myTargetInfo, vector<double>& result, himan::param p, int low, int high)
{

   	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

        result = h->VerticalMaximum(p,low,high);

}

void visibility::VertMax(shared_ptr<info> myTargetInfo, vector<double>& result, vector<himan::param> p, int low, int high)
{

   	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	result = h->VerticalMaximum(p,low,high);

}

void visibility::VertTMin(shared_ptr<info> myTargetInfo, vector<double>& result, int low, int high)
{

   	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

        result = h->VerticalMinimum(param("T-K"),low,high);

}

void visibility::VertFFValue(shared_ptr<info> myTargetInfo, vector<double>& result, vector<double>& value)
{

   	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

        result = h->VerticalValue(himan::param("FF-MS"), value);

}

