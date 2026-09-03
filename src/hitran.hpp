// This software does **not** include HITRAN data.

#pragma once

#include <complex>
#include <random>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <netcdf>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numbers>

#include "autodiff.hpp"

namespace hitran
{

class Molecule
{
	public:
		int molecule_id;
		std::string name;
		std::string general_formula;

		Molecule(void);
		Molecule(int, std::string, std::string);
};

class Isotopologue
{
	public:
		int global_id;
		int molecule_id;
		int local_iso_id;
		std::string isotopic_formula;
		int afgl_code;
		std::string general_formula;
		double abundance;
		double molar_mass;
		int gi;

		struct QTable
		{
			std::vector<double> T;
			std::vector<double> Q;
		};

		Isotopologue(void);
		Isotopologue(int, int, int, std::string, int, std::string, double, double, int);

		QTable qtable;
		void loadQTable(const netCDF::NcFile&);
		void loadQTable(const std::string&);
		void loadQTable_raw(const std::string&);
		template <typename T> T interpolateQ(T) const;
};

enum class BroadeningGas{Air, CO2, H2, He, H2O};

class Diluent
{
	public:
		double air;
		double CO2;
		double H2;
		double He;
		double H2O;
	
		Diluent(double, double, double, double, double);
		Diluent(void);
		void normalize(void);
};

template <typename T> struct BroadenerParameters
{
	T gamma;
	T n;
	T delta;
};

class Line
{
	private:
		template <typename T> BroadenerParameters<T> getBroadenerParameters(const BroadeningGas&) const;
		template <typename T> BroadenerParameters<T> getBroadenerParameters(const Diluent&) const;

		template <typename T> T computeLineStrengthT(T, const Isotopologue&) const;

		template <typename T> T computeLorentzHWHM(T, T, T, const BroadeningGas&) const;
		template <typename T> T computeLorentzHWHM(T, T, T, const Diluent&) const;

		template <typename T> T computePressureShift(T, const BroadeningGas&) const;
		template <typename T> T computePressureShift(T, const Diluent&) const;

		template <typename T> T computeDopplerHWHM(T, const Isotopologue&) const;

		template <typename T> T computeGaussianProfile(double, T, T) const;
		template <typename T> T computeLorentzProfile(double, T, T) const;

		template <typename T> T computeVoigtProfile(double, T, T, T, const Isotopologue&, const BroadeningGas&) const;
		template <typename T> T computeVoigtProfile(double, T, T, T, const Isotopologue&, const Diluent&) const;

	public:
		int molecular_id;
		int global_id;
		int local_id;

		double nu;
		double sw;
		double elower;
		double gamma_self;

		double gamma_air;
		double n_air;
		double delta_air;

		double gamma_CO2;
		double n_CO2;
		double delta_CO2;

		double gamma_H2;
		double n_H2;
		double delta_H2;

		double gamma_He;
		double n_He;
		double delta_He;

		double gamma_H2O;
		double n_H2O;
		double delta_H2O;
		
		template <typename T> T computeCrossSection(double, T, T, T, const Isotopologue&, const BroadeningGas&) const;
		template <typename T> T computeCrossSection(double, T, T, T, const Isotopologue&, const Diluent&) const;
};

class CIAGrid
{
	public:
		std::string molecule1;
		std::string molecule2;
		std::string ortho_para_ratio;
		std::vector<double> temperatures;
		std::vector<std::vector<double>> wavenumbers;
		std::vector<std::vector<double>> k_cia;

		void loadFromASCII(const std::string&);
		void loadCIA(const netCDF::NcGroup&);
		template <typename T> T evaluate(double, T) const;

	private:
		double interpolate1D(double, const std::vector<double>&, const std::vector<double>&) const;
};

class CIA
{
	private:
		std::vector<CIAGrid> loaded_grids;

		bool hasGroup(const netCDF::NcFile&, const std::string&) const;

	public:
		const std::vector<CIAGrid>& getLoadedGrids(void) const;
		void load(const netCDF::NcFile&, const std::vector<std::string>&);
		template <typename T> T computeAbsorptionCoefficient(double, T, const std::map<std::string, T>&) const;
};

inline const double AVOGADRO_CONSTANT = 6.02214076E23;
inline const double MOLAR_GAS_CONSTANT = 8.31446261815324;
inline const double PLANCK_CONSTANT = 6.62607015E-34;
inline const double SPEED_OF_LIGHT = 2.99792458E8;
inline const double BOLTZMANN_CONSTANT = 1.380649E-23;

const double ATM_TO_PA = 101325.0;
const double CM_TO_M = 1.0E-2;
const double UM_TO_M = 1.0E-6;
const double GAMMA_UNIT = 1.0 / (CM_TO_M * ATM_TO_PA);

const double TREF = 296.0; // K

inline const std::vector<Molecule> molecules ={
	Molecule(1, "H2O", "H2O"),
	Molecule(2, "CO2", "CO2"),
	Molecule(3, "O3", "O3"),
	Molecule(4, "N2O", "N2O"),
	Molecule(5, "CO", "CO"),
	Molecule(6, "CH4", "CH4"),
	Molecule(7, "O2", "O2"),
	Molecule(8, "NO", "NO"),
	Molecule(9, "SO2", "SO2"),
	Molecule(10, "NO2", "NO2"),
	Molecule(11, "NH3", "NH3"),
	Molecule(12, "HNO3", "HNO3"),
	Molecule(13, "OH", "OH"),
	Molecule(14, "HF", "HF"),
	Molecule(15, "HCl", "HCl"),
	Molecule(16, "HBr", "HBr"),
	Molecule(17, "HI", "HI"),
	Molecule(18, "ClO", "ClO"),
	Molecule(19, "OCS", "OCS"),
	Molecule(20, "H2CO", "H2CO"),
	Molecule(21, "HOCl", "HOCl"),
	Molecule(22, "N2", "N2"),
	Molecule(23, "HCN", "HCN"),
	Molecule(24, "CH3Cl", "CH3Cl"),
	Molecule(25, "H2O2", "H2O2"),
	Molecule(26, "C2H2", "C2H2"),
	Molecule(27, "C2H6", "C2H6"),
	Molecule(28, "PH3", "PH3"),
	Molecule(29, "COF2", "COF2"),
	Molecule(30, "SF6", "SF6"),
	Molecule(31, "H2S", "H2S"),
	Molecule(32, "HCOOH", "HCOOH"),
	Molecule(33, "HO2", "HO2"),
	Molecule(34, "O", "O"),
	Molecule(35, "ClONO2", "ClONO2"),
	Molecule(36, "NO+", "NO+"),
	Molecule(37, "HOBr", "HOBr"),
	Molecule(38, "C2H4", "C2H4"),
	Molecule(39, "CH3OH", "CH3OH"),
	Molecule(40, "CH3Br", "CH3Br"),
	Molecule(41, "CH3CN", "CH3CN"),
	Molecule(42, "CF4", "CF4"),
	Molecule(43, "C4H2", "C4H2"),
	Molecule(44, "HC3N", "HC3N"),
	Molecule(45, "H2", "H2"),
	Molecule(46, "CS", "CS"),
	Molecule(47, "SO3", "SO3"),
	Molecule(48, "C2N2", "C2N2"),
	Molecule(49, "COCl2", "COCl2"),
	Molecule(50, "SO", "SO"),
	Molecule(51, "CH3F", "CH3F"),
	Molecule(52, "GeH4", "GeH4"),
	Molecule(53, "CS2", "CS2"),
	Molecule(54, "CH3I", "CH3I"),
	Molecule(55, "NF3", "NF3"),
};

inline const std::vector<Isotopologue> isotopologues = {
	Isotopologue(1, 1, 1, "H216O", 161, "H2O", 9.97317E-01, 18.010565E-3, 1),
	Isotopologue(2, 1, 2, "H218O", 181, "H2O", 1.99983E-03, 20.014811E-3, 1),
	Isotopologue(3, 1, 3, "H217O", 171, "H2O", 3.71884E-04, 19.014780E-3, 6),
	Isotopologue(4, 1, 4, "HD16O", 162, "H2O", 3.10693E-04, 19.016740E-3, 6),
	Isotopologue(5, 1, 5, "HD18O", 182, "H2O", 6.23003E-07, 21.020985E-3, 6),
	Isotopologue(6, 1, 6, "HD17O", 172, "H2O", 1.15853E-07, 20.020956E-3, 36),
	Isotopologue(129, 1, 7, "D216O", 262, "H2O", 2.41974E-08, 20.022915E-3, 1),

	Isotopologue(7, 2, 1, "12C16O2", 626, "CO2", 9.84204E-01, 43.989830E-3, 1),
	Isotopologue(8, 2, 2, "13C16O2", 636, "CO2", 1.10574E-02, 44.993185E-3, 2),
	Isotopologue(9, 2, 3, "16O12C18O", 628, "CO2", 3.94707E-03, 45.994076E-3, 1),
	Isotopologue(10, 2, 4, "16O12C17O", 627, "CO2", 7.33989E-04, 44.994045E-3, 6),
	Isotopologue(11, 2, 5, "16O13C18O", 638, "CO2", 4.43446E-05, 46.997431E-3, 2),
	Isotopologue(12, 2, 6, "16O13C17O", 637, "CO2", 8.24623E-06, 45.997400E-3,12),
	Isotopologue(13, 2, 7, "12C18O2", 828, "CO2", 3.95734E-06, 47.998320E-3, 1),
	Isotopologue(14, 2, 8, "17O12C18O", 827, "CO2", 1.47180E-06, 46.998291E-3, 6),
	Isotopologue(121, 2, 9, "12C17O2", 727, "CO2", 1.36847E-07, 45.998262E-3, 1),
	Isotopologue(15, 2, 0, "13C18O2", 838, "CO2", 4.44600E-08, 49.001675E-3, 2),
	Isotopologue(120, 2, 11, "18O13C17O", 837, "CO2", 1.65354E-08, 48.001646E-3, 12),
	Isotopologue(122, 2, 12, "13C17O2", 737, "CO2", 1.53745E-09, 47.001618E-3, 2),

	Isotopologue(16, 3, 1, "16O3", 666, "O3", 9.92901E-01, 47.984745E-3, 1),
	Isotopologue(17, 3, 2, "16O16O18O", 668, "O3", 3.98194E-03, 49.988991E-3, 1),
	Isotopologue(18, 3, 3, "16O18O16O", 686, "O3", 1.99097E-03, 49.988991E-3, 1),
	Isotopologue(19, 3, 4, "16O16O17O", 667, "O3", 7.40475E-04, 48.988960E-3, 6),
	Isotopologue(20, 3, 5, "16O17O16O", 676, "O3", 3.70237E-04, 48.988960E-3, 6),

	Isotopologue(21, 4, 1, "14N216O", 446, "N2O", 9.90333E-01, 44.001062E-3, 9),
	Isotopologue(22, 4, 2, "14N15N16O", 456, "N2O", 3.64093E-03, 44.998096E-3, 6),
	Isotopologue(23, 4, 3, "15N14N16O", 546, "N2O", 3.64093E-03, 44.998096E-3, 6),
	Isotopologue(24, 4, 4, "14N218O", 448, "N2O", 1.98582E-03, 46.005308E-3, 9),
	Isotopologue(25, 4, 5, "14N217O", 447, "N2O", 3.69280E-04, 45.005278E-3, 54),

	Isotopologue(26, 5, 1, "12C16O", 26, "CO", 9.86544E-01, 27.994915E-3, 1),
	Isotopologue(27, 5, 2, "13C16O", 36, "CO", 1.10836E-02, 28.998270E-3, 2),
	Isotopologue(28, 5, 3, "12C18O", 28, "CO", 1.97822E-03, 29.999161E-3, 1),
	Isotopologue(29, 5, 4, "12C17O", 27, "CO", 3.67867E-04, 28.999130E-3, 6),
	Isotopologue(30, 5, 5, "13C18O", 38, "CO", 2.22250E-05, 31.002516E-3, 2),
	Isotopologue(31, 5, 6, "13C17O", 37, "CO", 4.13292E-06, 30.002485E-3, 12),

	Isotopologue(32, 6, 1, "12CH4", 211, "CH4", 9.88274E-01, 16.031300E-3, 1),
	Isotopologue(33, 6, 2, "13CH4", 311, "CH4", 1.11031E-02, 17.034655E-3, 2),
	Isotopologue(34, 6, 3, "12CH3D", 212, "CH4", 6.15751E-04, 17.037475E-3, 3),
	Isotopologue(35, 6, 4, "13CH3D", 312, "CH4", 6.91785E-06, 18.040830E-3, 6),

	Isotopologue(36, 7, 1, "16O2", 66, "O2", 9.95262E-01, 31.989830E-3, 1),
	Isotopologue(37, 7, 2, "16O18O", 68, "O2", 3.99141E-03, 33.994076E-3, 1),
	Isotopologue(38, 7, 3, "16O17O", 67, "O2", 7.42235E-04, 32.994045E-3, 6),

	Isotopologue(39, 8, 1, "14N16O", 46, "NO", 9.93974E-01, 29.997989E-3, 3),
	Isotopologue(40, 8, 2, "15N16O", 56, "NO", 3.65431E-03, 30.995023E-3, 2),
	Isotopologue(41, 8, 3, "14N18O", 48, "NO", 1.99312E-03, 32.002234E-3, 3),

	Isotopologue(42, 9, 1, "32S16O2", 626, "SO2", 9.45678E-01, 63.961901E-3, 1),
	Isotopologue(43, 9, 2, "34S16O2", 646, "SO2", 4.19503E-02, 65.957695E-3, 1),
	Isotopologue(137, 9, 3, "33S16O2", 636, "SO2", 7.46446E-03, 64.961286E-3, 4),
	Isotopologue(138, 9, 4, "16O32S18O", 628,"SO2", 3.79256E-03, 65.966146E-3, 1),

	Isotopologue(44, 10, 1, "14N16O2", 646, "NO2", 9.91616E-01, 45.992904E-3, 3),
	Isotopologue(130, 10, 2, "15N16O2", 656, "NO2", 3.64564E-03, 46.989938E-3, 2),

	Isotopologue(45, 11, 1, "14NH3", 4111, "NH3", 9.95872E-01, 17.026549E-3, 3),
	Isotopologue(46, 11, 2, "15NH3", 5111, "NH3", 3.66129E-03, 18.023583E-3, 2),

	Isotopologue(47, 12, 1, "H14N16O3", 146, "HNO3", 9.89110E-01, 62.995644E-3, 6),
	Isotopologue(117, 12, 2, "H15N16O3", 156, "HNO3", 3.63643E-03, 63.992678E-3, 4),

	Isotopologue(48, 13, 1, "16OH", 61, "OH", 9.97473E-01, 17.002740E-3, 2),
	Isotopologue(49, 13, 2, "18OH", 81, "OH", 2.00014E-03, 19.006986E-3, 2),
	Isotopologue(50, 13, 3, "16OD", 62, "OH", 1.55371E-04, 18.008915E-3, 3),

	Isotopologue(51, 14, 1, "H19F", 19, "HF", 9.99844E-01, 20.006229E-3, 4),
	Isotopologue(110, 14, 2, "D19F", 29, "HF", 1.55741E-04, 21.012404E-3, 6),

	Isotopologue(52, 15, 1, "H35Cl", 15, "HCl", 7.57587E-01, 35.976678E-3, 8),
	Isotopologue(53, 15, 2, "H37Cl", 17, "HCl", 2.42257E-01, 37.973729E-3, 8),
	Isotopologue(107, 15, 3, "D35Cl", 25, "HCl", 1.18005E-04, 36.982853E-3, 12),
	Isotopologue(108, 15, 4, "D37Cl", 27, "HCl", 3.77350E-05, 38.979904E-3, 12),

	Isotopologue(54, 16, 1, "H79Br", 19, "HBr", 5.06781E-01, 79.926160E-3, 8),
	Isotopologue(55, 16, 2, "H81Br", 11, "HBr", 4.93063E-01, 81.924115E-3, 8),
	Isotopologue(111, 16, 3, "D79Br", 29, "HBr", 7.89384E-05, 80.932336E-3, 12),
	Isotopologue(112, 16, 4, "D81Br", 21, "HBr", 7.68016E-05, 82.930289E-3, 12),

	Isotopologue(56, 17, 1, "H127I", 17, "HI", 9.99844E-01,127.912297E-3, 12),
	Isotopologue(113, 17, 2, "D127I", 27, "HI", 1.55741E-04,128.918472E-3, 18),

	Isotopologue(57, 18, 1, "35Cl16O", 56, "ClO", 7.55908E-01, 50.963768E-3, 4),
	Isotopologue(58, 18, 2, "37Cl16O", 76, "ClO", 2.41720E-01, 52.960819E-3, 4),

	Isotopologue(59, 19, 1, "16O12C32S", 622, "OCS", 9.37395E-01, 59.966986E-3, 1),
	Isotopologue(60, 19, 2, "16O12C34S", 624, "OCS", 4.15828E-02, 61.962780E-3, 1),
	Isotopologue(61, 19, 3, "16O13C32S", 632, "OCS", 1.05315E-02, 60.970341E-3, 2),
	Isotopologue(62, 19, 4, "16O12C33S", 623, "OCS", 7.39908E-03, 60.966371E-3, 4),
	Isotopologue(63, 19, 5, "18O12C32S", 822, "OCS", 1.87967E-03, 61.971231E-3, 1),
	Isotopologue(135, 19, 6, "16O13C34S", 634, "OCS", 4.67176E-04, 62.966137E-3, 2),

	Isotopologue(64, 20, 1, "H212C16O", 126, "H2CO", 9.86237E-01, 30.010565E-3, 1),
	Isotopologue(65, 20, 2, "H213C16O", 136, "H2CO", 1.10802E-02, 31.013920E-3, 2),
	Isotopologue(66, 20, 3, "H212C18O", 128, "H2CO", 1.97761E-03, 32.014811E-3, 1),

	Isotopologue(67, 21, 1, "H16O35Cl", 165, "HOCl", 7.55790E-01, 51.971593E-3, 8),
	Isotopologue(68, 21, 2, "H16O37Cl", 167, "HOCl", 2.41683E-01, 53.968644E-3, 8),

	Isotopologue(69, 22, 1, "14N2", 44, "N2", 9.92687E-01, 28.006148E-3, 1),
	Isotopologue(118, 22, 2, "14N15N", 45, "N2", 7.29916E-03, 29.003182E-3, 6),

	Isotopologue(70, 23, 1, "H12C14N", 124, "HCN", 9.85114E-01, 27.010899E-3, 6),
	Isotopologue(71, 23, 2, "H13C14N", 134, "HCN", 1.10676E-02, 28.014254E-3, 12),
	Isotopologue(72, 23, 3, "H12C15N", 125, "HCN", 3.62174E-03, 28.007933E-3, 4),

	Isotopologue(73, 24, 1, "12CH335Cl", 215, "CH3Cl", 7.48937E-01, 49.992328E-3, 4),
	Isotopologue(74, 24, 2, "12CH337Cl", 217, "CH3Cl", 2.39491E-01, 51.989379E-3, 4),

	Isotopologue(75, 25, 1, "H216O2", 1661,"H2O2", 9.94952E-01, 34.005480E-3, 1),

	Isotopologue(76, 26, 1, "12C2H2", 1221, "C2H2", 9.77599E-01, 26.015650E-3, 1),
	Isotopologue(77, 26, 2, "H12C13CH", 1231, "C2H2", 2.19663E-02, 27.019005E-3, 8),
	Isotopologue(105, 26, 3, "H12C12CD", 1222, "C2H2", 3.04550E-04, 27.021825E-3, 6),

	Isotopologue(78, 27, 1, "12C2H6", 1221, "C2H6", 9.76990E-01, 30.046950E-3, 1),
	Isotopologue(106, 27, 2, "12CH313CH3", 1231, "C2H6", 2.19526E-02, 31.050305E-3, 2),

	Isotopologue(79, 28, 1, "31PH3", 1111, "PH3", 9.99533E-01, 33.997241E-3, 2),

	Isotopologue(80, 29, 1, "12C16O19F2", 269, "COF2", 9.86544E-01, 65.991722E-3, 1),
	Isotopologue(119, 29, 2, "13C16O19F2", 369, "COF2", 1.10837E-02, 66.995078E-3, 2),

	Isotopologue(126, 30, 1, "32S19F6", 29, "SF6", 9.50180E-01,145.962494E-3, 1),

	Isotopologue(81, 31, 1, "H232S", 121, "H2S", 9.49884E-01, 33.987721E-3, 1),
	Isotopologue(82, 31, 2, "H234S", 141, "H2S", 4.21369E-02, 35.983515E-3, 1),
	Isotopologue(83, 31, 3, "H233S", 131, "H2S", 7.49766E-03, 34.987105E-3, 4),

	Isotopologue(84, 32, 1, "H12C16O16OH", 126, "HCOOH", 9.83898E-01, 46.005480E-3, 4),

	Isotopologue(85, 33, 1, "H16O2", 166, "HO2", 9.95107E-01, 32.997655E-3, 2),

	Isotopologue(86, 34, 1, "16O", 6, "O", 9.97628E-01, 15.994915E-3, 1),

	Isotopologue(127, 35, 1, "35Cl16O14N16O2", 5646, "ClONO2", 7.49570E-01, 96.956672E-3, 12),
	Isotopologue(128, 35, 2, "37Cl16O14N16O2", 7646, "ClONO2", 2.39694E-01, 98.953723E-3, 12),

	Isotopologue(87, 36, 1, "14N16O+", 46, "NO+", 9.93974E-01, 29.997989E-3, 3),

	Isotopologue(88, 37, 1, "H16O79Br", 169, "HOBr", 5.05579E-01, 95.921076E-3, 8),
	Isotopologue(89, 37, 2, "H16O81Br", 161, "HOBr", 4.91894E-01, 97.919030E-3, 8),

	Isotopologue(90, 38, 1, "12C2H4", 221, "C2H4", 9.77294E-01, 28.031300E-3, 1),
	Isotopologue(91, 38, 2, "12CH213CH2", 231, "C2H4", 2.19595E-02, 29.034655E-3, 2),

	Isotopologue(92, 39, 1, "12CH316OH", 2161, "CH3OH", 9.85930E-01, 32.026215E-3, 2),

	Isotopologue(93, 40, 1, "12CH379Br", 219, "CH3Br", 5.00995E-01, 93.941811E-3, 4),
	Isotopologue(94, 40, 2, "12CH381Br", 211, "CH3Br", 4.87433E-01, 95.939764E-3, 4),

	Isotopologue(95, 41, 1, "12CH312C14N", 2124, "CH3CN", 9.73866E-01, 41.026549E-3, 3),

	Isotopologue(96, 42, 1, "12C19F4", 29, "CF4", 9.88890E-01, 87.993616E-3, 1),

	Isotopologue(116,43,1, "12C4H2", 2211, "C4H2", 9.55998E-01, 50.015650E-3, 1),

	Isotopologue(109, 44, 1, "H12C314N", 1224, "HC3N", 9.63346E-01, 51.010899E-3, 6),

	Isotopologue(103, 45, 1, "H2", 11, "H2", 9.99688E-01,  2.015650E-3, 1),
	Isotopologue(115, 45, 2, "HD", 12, "H2", 3.11432E-04,  3.021825E-3, 6),

	Isotopologue(97, 46, 1, "12C32S", 22, "CS", 9.39624E-01, 43.972070E-3, 1),
	Isotopologue(98, 46, 2, "12C34S", 24, "CS", 4.16817E-02, 45.967866E-3, 1),
	Isotopologue(99, 46, 3, "13C32S", 32, "CS", 1.05565E-02, 44.975425E-3, 2),
	Isotopologue(100, 46, 4, "12C33S", 23, "CS", 7.41668E-03, 44.971456E-3, 4),

	Isotopologue(114, 47, 1, "32S16O3", 26, "SO3", 9.43434E-01, 79.956815E-3, 1),

	Isotopologue(123, 48, 1, "12C214N2", 4224, "C2N2", 9.70752E-01, 52.006148E-3, 1),

	Isotopologue(124, 49, 1, "12C16O35Cl2", 2655, "COCl2", 5.66392E-01, 97.932620E-3, 1),
	Isotopologue(125, 49, 2, "12C16O35Cl37Cl", 2657, "COCl2", 3.62235E-01, 99.929672E-3, 16),

	Isotopologue(146, 50, 1, "32S16O", 26, "SO", 9.47926E-01, 47.966986E-3, 1),
	Isotopologue(147, 50, 2, "34S16O", 46, "SO", 4.20500E-02, 49.962782E-3, 1),
	Isotopologue(148, 50, 3, "32S18O", 28, "SO", 1.90079E-03, 49.971231E-3, 1),

	Isotopologue(144,51,1, "12CH319F", 219, "CH3F", 9.88428E-01, 34.021878E-3, 2),

	Isotopologue(139, 52, 1, "74GeH4", 411, "GeH4", 3.65172E-01, 77.952479E-3, 1),
	Isotopologue(140, 52, 2, "72GeH4", 211, "GeH4", 2.74129E-01, 75.953380E-3, 1),
	Isotopologue(141, 52, 3, "70GeH4", 011, "GeH4", 2.05072E-01, 73.955550E-3, 1),
	Isotopologue(142, 52, 4, "73GeH4", 311, "GeH4", 7.75517E-02, 76.954764E-3, 10),
	Isotopologue(143, 52, 5, "76GeH4", 611, "GeH4", 7.75517E-02, 79.952703E-3, 1),

	Isotopologue(131, 53, 1, "12C32S2", 222, "CS2", 8.92811E-01, 75.944140E-3, 1),
	Isotopologue(132, 53, 2, "32S12C34S", 224, "CS2", 7.92103E-02, 77.939936E-3, 1),
	Isotopologue(133, 53, 3, "32S12C33S", 223, "CS2", 1.40944E-02, 76.943526E-3, 4),
	Isotopologue(134, 53, 4, "13C32S2", 232, "CS2", 1.00306E-02, 76.947495E-3, 2),

	Isotopologue(145, 54, 1, "12CH3127I", 217, "CH3I", 9.88428E-01, 141.927947E-3, 6),

	Isotopologue(136, 55, 1, "14N19F3", 4999, "NF3", 9.96337E-01, 70.998286E-3, 3)
};

std::vector<std::string> splitString(std::string, char);
int molecule_id_from_name(std::string);
std::string name_from_molecule_id(int);
std::vector<Isotopologue> isos_for_molecule(int);
std::vector<int> mol_local_from_global(int);
int global_from_mol_local(int, int);
std::vector<Line> loadLines(const netCDF::NcFile&, const Isotopologue&, double, double, bool);
std::vector<Line> loadLines(const std::string&, const Isotopologue&, double, double, bool);
std::vector<std::string> listFiles(std::string, std::string, bool, bool);
std::vector<Line> loadLines_raw(const std::string&, const Isotopologue&, double, double, bool);
std::vector<Line> loadLines_raw(const std::string&, int, std::vector<int>, double, double, bool);
std::vector<std::string> readCIAHeader_raw(std::string);
void generateNetCDF(const std::string&, const std::string&);

inline std::vector<std::string> splitString(std::string str, char del)
{
	int first = 0;
	int last = str.find_first_of(del);

	if(last != std::string::npos)
	{
		std::vector<std::string> result;

		while (first < str.size())
		{
			std::string subStr(str, first, last - first);

			result.push_back(subStr);

			first = last + 1;
			last = str.find_first_of(del, first);

			if (last == std::string::npos)
			{
				last = str.size();
			}
		}

		return result;
	}
	else
	{
		std::vector<std::string> result = {str};

		return result;
	}
}

inline int molecule_id_from_name(std::string name)
{
	for (auto& m : molecules)
	{
		if (name == m.name)
		{
			return m.molecule_id;
		}
	}

	throw std::runtime_error("Molecule not found: " + name);
}

inline std::string name_from_molecule_id(int id)
{
	for (auto& m : molecules)
	{
		if (m.molecule_id == id)
		{
			return std::string(m.name);
		}
	}

	throw std::runtime_error("Molecule not found: " + std::to_string(id));
}

inline std::vector<Isotopologue> isos_for_molecule(int mol_id)
{
	std::vector<Isotopologue> result;
	
	for (auto& iso : isotopologues)
	{
		if (iso.molecule_id == mol_id)
		{
			result.push_back(iso);
		}
	}

	return result;
}

inline std::vector<int> mol_local_from_global(int global_id)
{
	for (auto& iso : isotopologues)
	{
		if (iso.global_id == global_id)
		{
			std::vector<int> result = {iso.molecule_id, iso.local_iso_id};
			return result;
		}
	}
	
	throw std::runtime_error("Isotopologue not found: " + std::to_string(global_id));
}

inline int global_from_mol_local(int mol_id, int local_id)
{
	for (auto& iso : isotopologues)
	{
		if (iso.molecule_id == mol_id && iso.local_iso_id == local_id)
		{
			return iso.global_id;
		}
	}

	throw std::runtime_error("Isotopologue not found: (" + std::to_string(mol_id) + ", " + std::to_string(local_id) + ")");
}

inline Molecule::Molecule(void)
{
	return;
}

inline Molecule::Molecule(int id, std::string n, std::string formula)
{
	molecule_id = id;
	name = n;
	general_formula = formula;

	return;
}

inline Isotopologue::Isotopologue(void)
{
	return;
}

inline Isotopologue::Isotopologue(int gid, int mid, int lid, std::string formula, int code, std::string gformula, double abd, double mass, int g)
{
	global_id = gid;
	molecule_id = mid;
	local_iso_id = lid;
	isotopic_formula = formula;
	afgl_code = code;
	general_formula = gformula;
	abundance = abd;
	molar_mass = mass;
	gi = g;

	return;
}

inline const Isotopologue iso_from_isotopic_formula(std::string iso_formula)
{
	for (auto& iso : isotopologues)
	{
		if (iso.isotopic_formula == iso_formula)
		{
			return iso;
		}
	}
	
	throw std::runtime_error("Isotopologue not found: " + iso_formula);
}

inline const Isotopologue iso_from_global(int global_id)
{
	for (auto& iso : isotopologues)
	{
		if (iso.global_id == global_id)
		{
			return iso;
		}
	}
	
	throw std::runtime_error("Isotopologue not found: " + std::to_string(global_id));
}

inline std::string general_from_isotopic_formula(std::string iso_formula)
{
	return iso_from_isotopic_formula(iso_formula).general_formula;
}

inline void Isotopologue::loadQTable(const netCDF::NcFile& nc)
{
	netCDF::NcGroup g = nc.getGroup("line_by_line").getGroup(std::to_string(molecule_id)).getGroup(std::to_string(global_id)).getGroup("partition_function");

	netCDF::NcDim dim_T = g.getDim("temperature");
	int n_T = dim_T.getSize();

	netCDF::NcVar var_T = g.getVar("T");
	qtable.T.resize(n_T);
	var_T.getVar(qtable.T.data());

	netCDF::NcVar var_Q = g.getVar("Q");	
	qtable.Q.resize(n_T);
	var_Q.getVar(qtable.Q.data());

	return;
}

inline void Isotopologue::loadQTable(const std::string& filename)
{
	netCDF::NcFile nc(filename, netCDF::NcFile::read);

	loadQTable(nc);

	return;
}

inline void Isotopologue::loadQTable_raw(const std::string& filename)
{
	std::ifstream input(filename);

	qtable.T.clear();
	qtable.Q.clear();

	if (!input.is_open())
	{
		throw std::runtime_error("Cannot open Q(T) file: " + filename);
	}
	
	double q, t;

	while(input >> t >> q)
	{
		qtable.T.push_back(t);
		qtable.Q.push_back(q);
	}

	return;
}

template <typename T> inline T Isotopologue::interpolateQ(T temperature) const
{
	const auto& T_vec = qtable.T;
	const auto& Q_vec = qtable.Q;

	double vt = autodiff::get_value(temperature);

	if(vt < T_vec[0] || T_vec[T_vec.size() - 1] < vt)
	{
		throw std::runtime_error("temperature: " + std::to_string(vt) + " is out of range.");
	}

	auto it = std::lower_bound(T_vec.begin(), T_vec.end(), vt);
	int hi = std::distance(T_vec.begin(), it);
	int lo = hi - 1;

	double t0 = T_vec[lo];
	double t1 = T_vec[hi];
	double q0 = Q_vec[lo];
	double q1 = Q_vec[hi];

	T r = (temperature - T(t0)) / T(t1 - t0);
	
	return T(q0) + r * T(q1 - q0);
}

inline Diluent::Diluent(double q_air, double q_co2, double q_h2, double q_he, double q_h2o)
{
	air = q_air;
	CO2 = q_co2;
	H2 = q_h2;
	He = q_he;
	H2O = q_h2o;

	normalize();
}

inline Diluent::Diluent(void)
{
	return;
}

inline void Diluent::normalize(void)
{
	double sum = air + CO2 + H2 + He + H2O;
	
	if(sum <= 0.0)
	{
		throw std::runtime_error("Diluent sum is zero/non-positive.");
	}

	air /= sum;
	CO2 /= sum;
	H2 /= sum;
	He /= sum;
	H2O /= sum;

	return;
}

template <typename T> inline BroadenerParameters<T> Line::getBroadenerParameters(const BroadeningGas& broadening_gas) const
{
	BroadenerParameters<T> params;

	if(broadening_gas == BroadeningGas::Air)
	{
		params.gamma = gamma_air;
		params.n = n_air;
		params.delta = delta_air;
	}
	else if(broadening_gas == BroadeningGas::CO2)
	{
		params.gamma = gamma_CO2;
		params.n = n_CO2;
		params.delta = delta_CO2;
	}
	else if(broadening_gas == BroadeningGas::H2)
	{
		params.gamma = gamma_H2;
		params.n = n_H2;
		params.delta = delta_H2;
	}
	else if(broadening_gas == BroadeningGas::He)
	{
		params.gamma = gamma_He;
		params.n = n_He;
		params.delta = delta_He;
	}
	else if(broadening_gas == BroadeningGas::H2O)
	{
		params.gamma = gamma_H2O;
		params.n = n_H2O;
		params.delta = delta_H2O;
	}

	return params;
}

template <typename T> inline BroadenerParameters<T> Line::getBroadenerParameters(const Diluent& diluent) const
{
	// to be fixed 

	BroadenerParameters<T> params;

	params.gamma = diluent.air * gamma_air + diluent.CO2 * (std::isnan(gamma_CO2) ? gamma_air : gamma_CO2) + diluent.H2 * (std::isnan(gamma_H2) ? gamma_air : gamma_H2) + diluent.He * (std::isnan(gamma_He) ? gamma_air : gamma_He) + diluent.H2O * (std::isnan(gamma_H2O) ? gamma_air : gamma_H2O);
	params.n = diluent.air * n_air + diluent.CO2 * (std::isnan(n_CO2) ? n_air : n_CO2) + diluent.H2 * (std::isnan(n_H2) ? n_air : n_H2) + diluent.He * (std::isnan(n_He) ? n_air : n_He) + diluent.H2O * (std::isnan(n_H2O) ? n_air : n_H2O);
	params.delta = diluent.air * delta_air + diluent.CO2 * (std::isnan(delta_CO2) ? 0.0 : delta_CO2) + diluent.H2 * (std::isnan(delta_H2) ? 0.0 : delta_H2) + diluent.He * (std::isnan(delta_He) ? 0.0 : delta_He) + diluent.H2O * (std::isnan(delta_H2O) ? 0.0 : delta_H2O);

	return params;
}

template <typename T> inline T Line::computeLineStrengthT(T temperature, const Isotopologue& iso) const
{
	using std::exp; using autodiff::exp;

	const double c2 = PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT;
	T exponent = exp(T(-elower * c2) * (T(1.0) / temperature - T(1.0 / TREF)));
	T num = T(1.0) - exp(T(-c2 * nu) / temperature);
	double den = 1.0 - std::exp(- c2 * nu / TREF);

	T q_ratio = T(iso.interpolateQ(TREF)) / iso.interpolateQ(temperature);
	return T(sw) * q_ratio * exponent * (num / T(den));
}

template <typename T> inline T Line::computeDopplerHWHM(T temperature, const Isotopologue& iso) const
{
	using std::sqrt; using autodiff::sqrt;
	using std::log; using autodiff::log;

	const T alpha = T(nu / SPEED_OF_LIGHT) * sqrt(T(2.0 * AVOGADRO_CONSTANT * BOLTZMANN_CONSTANT * std::log(2.0) / iso.molar_mass) * temperature);
	return alpha;
}

template <typename T> inline T Line::computeLorentzHWHM(T temperature, T pressure_self, T pressure, const BroadeningGas& broadening_gas) const
{
	using std::pow; using autodiff::pow;
	const auto p = getBroadenerParameters<T>(broadening_gas);
	return pow(T(TREF) / temperature, p.n) * (gamma_self * pressure_self + p.gamma * (pressure - pressure_self));
}

template <typename T> inline T Line::computeLorentzHWHM(T temperature, T pressure_self, T pressure, const Diluent& diluent) const
{
	using std::pow; using autodiff::pow;
	const auto p = getBroadenerParameters<T>(diluent);
	return pow(T(TREF) / temperature, p.n) * (gamma_self * pressure_self + p.gamma * (pressure - pressure_self));
}

template <typename T> inline T Line::computePressureShift(T pressure, const BroadeningGas& broadening_gas) const
{
	const auto p = getBroadenerParameters<T>(broadening_gas);
	return p.delta * pressure;
}

template <typename T> inline T Line::computePressureShift(T pressure, const Diluent& diluent) const
{
	const auto p = getBroadenerParameters<T>(diluent);
	return p.delta * pressure;
}

template <typename T> inline T Line::computeGaussianProfile(double wavenumber, T wavenumber_center, T alphaD) const
{
	using std::exp; using autodiff::exp;

	const T dnu = T(wavenumber) - wavenumber_center;

	return T(std::sqrt(std::numbers::ln2)) * T(std::numbers::inv_sqrtpi) / alphaD * exp(T(-std::numbers::ln2) * (dnu * dnu) / (alphaD * alphaD));
}

template <typename T> inline T Line::computeLorentzProfile(double wavenumber, T wavenumber_center, T gammaL) const
{
	const T dnu = T(wavenumber) - wavenumber_center;

	return (gammaL / (dnu * dnu + gammaL * gammaL)) * T(std::numbers::inv_pi);
}

template <typename T> inline T Line::computeVoigtProfile(double wavenumber, T temperature, T pressure_self, T pressure, const Isotopologue& iso, const BroadeningGas& broadening_gas) const
{
	T wavenumber_center = T(nu) + computePressureShift(pressure, broadening_gas);

	T alphaD = computeDopplerHWHM(temperature, iso);
	T gammaL = computeLorentzHWHM(temperature, pressure_self, pressure, broadening_gas);

	if (autodiff::get_value(alphaD) <= 0.0)
	{
		return computeLorentzProfile(wavenumber, wavenumber_center, gammaL);
	}
	else if (autodiff::get_value(gammaL) <= 0.0)
	{
		return computeGaussianProfile(wavenumber, wavenumber_center, alphaD);
	}
	else
	{
		using std::sqrt; using autodiff::sqrt;
		using std::log; using autodiff::log;
		using std::exp; using autodiff::exp;

		T s = sqrt(T(std::numbers::ln2));
		T x = s * (T(wavenumber) - wavenumber_center) / alphaD;
		T y = s * gammaL / alphaD;

		autodiff::complex<T> z(x, y);
		auto w = autodiff::faddeeva(z);

		T Vxy = w.real() * T(std::numbers::inv_sqrtpi);

		if(autodiff::get_value(y) < 1.0E-10)
		{
			const T eta = y / (y + T(1.0E-10));
			const T gauss = exp(- x * x) * T(std::numbers::inv_sqrtpi);
			Vxy = eta * Vxy + (T(1.0) - eta) * gauss;
		}

		return s / alphaD * Vxy;
	}
}

template <typename T> inline T Line::computeVoigtProfile(double wavenumber, T temperature, T pressure_self, T pressure, const Isotopologue& iso, const Diluent& diluent) const
{
	T wavenumber_center = T(nu) + computePressureShift(pressure, diluent);

	T alphaD = computeDopplerHWHM(temperature, iso);
	T gammaL = computeLorentzHWHM(temperature, pressure_self, pressure, diluent);

	if (autodiff::get_value(alphaD) <= 0.0)
	{
		return computeLorentzProfile(wavenumber, wavenumber_center, gammaL);
	}
	else if (autodiff::get_value(gammaL) <= 0.0)
	{
		return computeGaussianProfile(wavenumber, wavenumber_center, alphaD);
	}
	else
	{
		using std::sqrt; using autodiff::sqrt;
		using std::log; using autodiff::log;
		using std::exp; using autodiff::exp;

		T s = sqrt(T(std::numbers::ln2));
		T x = s * (T(wavenumber) - wavenumber_center) / alphaD;
		T y = s * gammaL / alphaD;

		autodiff::complex<T> z(x, y);
		auto w = autodiff::faddeeva(z);

		T Vxy = w.real() * T(std::numbers::inv_sqrtpi);

		if(autodiff::get_value(y) < 1.0E-10)
		{
			const T eta = y / (y + T(1.0E-10));
			const T gauss = exp(- x * x) * T(std::numbers::inv_sqrtpi);
			Vxy = eta * Vxy + (T(1.0) - eta) * gauss;
		}

		return s / alphaD * Vxy;
	}
}

template <typename T> inline T Line::computeCrossSection(double wavenumber, T temperature, T pressure_self, T pressure, const Isotopologue& iso, const BroadeningGas& broadening_gas) const
{
	T S = computeLineStrengthT(temperature, iso);
	T f = computeVoigtProfile(wavenumber, temperature, pressure_self, pressure, iso, broadening_gas);

	return S * f;
}

template <typename T> inline T Line::computeCrossSection(double wavenumber, T temperature, T pressure_self, T pressure, const Isotopologue& iso, const Diluent& diluent) const
{
	T S = computeLineStrengthT(temperature, iso);
	T f = computeVoigtProfile(wavenumber, temperature, pressure_self, pressure, iso, diluent);

	return S * f;
}

inline std::vector<Line> loadLines(const netCDF::NcFile& nc, const Isotopologue& isotopologue, double nu_min, double nu_max, bool weight_by_abundance = false)
{
	netCDF::NcGroup g = nc.getGroup("line_by_line").getGroup(std::to_string(isotopologue.molecule_id)).getGroup(std::to_string(isotopologue.global_id)).getGroup("lines");

	netCDF::NcDim dim_line = g.getDim("line");
	int n_line = dim_line.getSize();

	netCDF::NcVar var_nu = g.getVar("wavenumber");	

	std::vector<double> wavenumber(n_line);
	var_nu.getVar(wavenumber.data());

	auto it0 = std::lower_bound(wavenumber.begin(), wavenumber.end(), nu_min);
	auto it1 = std::upper_bound(wavenumber.begin(), wavenumber.end(), nu_max);

	if (it0 == wavenumber.end() || it0 >= it1)
	{
		return {};
	}

	size_t start_idx = size_t(it0 - wavenumber.begin());
	size_t cnt = size_t(it1 - it0);

	wavenumber.clear();
	wavenumber.resize(cnt);
	var_nu.getVar({start_idx}, {cnt}, wavenumber.data());

	netCDF::NcVar var_S = g.getVar("S");
	std::vector<double> S(cnt);
	var_S.getVar({start_idx}, {cnt}, S.data());

	netCDF::NcVar var_elower = g.getVar("elower");
	std::vector<double> elower(cnt);
	var_elower.getVar({start_idx}, {cnt}, elower.data());

	netCDF::NcVar var_gamma_self = g.getVar("gamma_self");
	std::vector<double> gamma_self(cnt);
	var_gamma_self.getVar({start_idx}, {cnt}, gamma_self.data());

	netCDF::NcVar var_gamma_air = g.getVar("gamma_air");
	std::vector<double> gamma_air(cnt);
	var_gamma_air.getVar({start_idx}, {cnt}, gamma_air.data());

	netCDF::NcVar var_n_air = g.getVar("n_air");
	std::vector<double> n_air(cnt);
	var_n_air.getVar({start_idx}, {cnt}, n_air.data());

	netCDF::NcVar var_delta_air = g.getVar("delta_air");
	std::vector<double> delta_air(cnt);
	var_delta_air.getVar({start_idx}, {cnt}, delta_air.data());

	netCDF::NcVar var_gamma_CO2 = g.getVar("gamma_CO2");
	std::vector<double> gamma_CO2(cnt);
	var_gamma_CO2.getVar({start_idx}, {cnt}, gamma_CO2.data());

	netCDF::NcVar var_n_CO2 = g.getVar("n_CO2");
	std::vector<double> n_CO2(cnt);
	var_n_CO2.getVar({start_idx}, {cnt}, n_CO2.data());

	netCDF::NcVar var_delta_CO2 = g.getVar("delta_CO2");
	std::vector<double> delta_CO2(cnt);
	var_delta_CO2.getVar({start_idx}, {cnt}, delta_CO2.data());

	netCDF::NcVar var_gamma_H2 = g.getVar("gamma_H2");
	std::vector<double> gamma_H2(cnt);
	var_gamma_H2.getVar({start_idx}, {cnt}, gamma_H2.data());

	netCDF::NcVar var_n_H2 = g.getVar("n_H2");
	std::vector<double> n_H2(cnt);
	var_n_H2.getVar({start_idx}, {cnt}, n_H2.data());

	netCDF::NcVar var_delta_H2 = g.getVar("delta_H2");
	std::vector<double> delta_H2(cnt);
	var_delta_H2.getVar({start_idx}, {cnt}, delta_H2.data());

	netCDF::NcVar var_gamma_He = g.getVar("gamma_He");
	std::vector<double> gamma_He(cnt);
	var_gamma_He.getVar({start_idx}, {cnt}, gamma_He.data());

	netCDF::NcVar var_n_He = g.getVar("n_He");
	std::vector<double> n_He(cnt);
	var_n_He.getVar({start_idx}, {cnt}, n_He.data());

	netCDF::NcVar var_delta_He = g.getVar("delta_He");
	std::vector<double> delta_He(cnt);
	var_delta_He.getVar({start_idx}, {cnt}, delta_He.data());

	netCDF::NcVar var_gamma_H2O = g.getVar("gamma_H2O");
	std::vector<double> gamma_H2O(cnt);
	var_gamma_H2O.getVar({start_idx}, {cnt}, gamma_H2O.data());

	netCDF::NcVar var_n_H2O = g.getVar("n_H2O");
	std::vector<double> n_H2O(cnt);
	var_n_H2O.getVar({start_idx}, {cnt}, n_H2O.data());

	netCDF::NcVar var_delta_H2O = g.getVar("delta_H2O");
	std::vector<double> delta_H2O(cnt);
	var_delta_H2O.getVar({start_idx}, {cnt}, delta_H2O.data());

	std::vector<Line> out(cnt);

	for(int i = 0; i < cnt; i ++)
	{
		Line line;
		line.nu = wavenumber[i];
		line.sw = S[i];

		if(weight_by_abundance == false)
		{
			line.sw /= isotopologue.abundance;
		}

		line.elower = elower[i];
		line.gamma_self = gamma_self[i];
		
		line.gamma_air = gamma_air[i];
		line.n_air = n_air[i];
		line.delta_air = delta_air[i];

		line.gamma_CO2 = gamma_CO2[i];
		line.n_CO2 = n_CO2[i];
		line.delta_CO2 = delta_CO2[i];

		line.gamma_H2 = gamma_H2[i];
		line.n_H2 = n_H2[i];
		line.delta_H2 = delta_H2[i];

		line.gamma_He = gamma_He[i];
		line.n_He = n_He[i];
		line.delta_He = delta_He[i];

		line.gamma_H2O = gamma_H2O[i];
		line.n_H2O = n_H2O[i];
		line.delta_H2O = delta_H2O[i];
		
		out[i] = line;
	}

	return out;
}

inline std::vector<Line> loadLines(const std::string& filename, const Isotopologue& isotopologue, double nu_min, double nu_max, bool weight_by_abundance = false)
{
	netCDF::NcFile nc(filename, netCDF::NcFile::read);

	return loadLines(nc, isotopologue, nu_min, nu_max, weight_by_abundance);
}

inline std::vector<std::string> listFiles(std::string directory, std::string extension, bool return_fullpath = false, bool search_recursive = false)
{
	if (extension.empty())
	{
		throw std::runtime_error("Extension is not specified.");
	}
	if(extension[0] != '.')
	{
		extension = "." + extension;
	}

	auto to_lower = [](std::string s)
	{
		std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::tolower(c);});
		return s;
	};

	const std::string ext_lower = to_lower(extension);

	std::filesystem::path dirpath(directory);
	if(!std::filesystem::exists(dirpath))
	{
		throw std::runtime_error("Directory does not exist: " + directory);
	}
	if(!std::filesystem::is_directory(dirpath))
	{
		throw std::runtime_error("Not a directory: " + directory);
	}

	std::vector<std::string> out;
	auto opts = std::filesystem::directory_options::skip_permission_denied;

	auto consider = [&](const std::filesystem::directory_entry& e)
	{
		if(!e.is_regular_file())
		{
			return;
		}

		std::string eext = to_lower(e.path().extension().string());

		if(eext == ext_lower)
		{
			out.push_back(return_fullpath ? e.path().string() : e.path().filename().string());
		}
	};

	try
	{
		if (search_recursive)
		{
			for(const auto& e : std::filesystem::recursive_directory_iterator(dirpath, opts))
			{
				consider(e);
			}
		}
		else
		{
			for(const auto& e : std::filesystem::directory_iterator(dirpath, opts))
			{
				consider(e);
			}
		}
	}
	catch(const std::exception& ex)
	{
		std::cerr << "Warning: " << ex.what() << '\n';
	}

	std::sort(out.begin(), out.end());

	out.erase(std::unique(out.begin(), out.end()), out.end());

	return out;
}

inline std::vector<Line> loadLines_raw(const std::string& filename, const Isotopologue &isotopologue, double nu_min, double nu_max, bool header = true)
{
	if (nu_max < nu_min)
	{
		std::swap(nu_min, nu_max);
	}

	std::ifstream fin(filename);
	if (!fin.is_open())
	{
		throw std::runtime_error("Cannot open line file: " + filename);
	}

	std::vector<Line> out;
	std::string line;

	if(header == true)
	{
		std::getline(fin, line);
	}

	while (std::getline(fin, line))
	{
		auto data = splitString(line, ',');

		if (data.size() != 54)
		{
			throw std::runtime_error("Invalid format: " + filename);
		}

		for(int i = 0; i < data.size(); i ++)
		{
			if(data[i] == "#")
			{
				data[i] = std::to_string(std::numeric_limits<double>::quiet_NaN());
			}
		}

		int molec_id = std::stoi(data[0]);
		int local_iso_id = std::stoi(data[1]);
		if (molec_id != isotopologue.molecule_id || local_iso_id != isotopologue.local_iso_id)
		{
			continue;
		}

		double nu = std::stod(data[2]) / CM_TO_M;
		if (nu < nu_min || nu_max < nu)
		{
			continue;
		}

		double sw = std::stod(data[3]) * CM_TO_M;
		double elower = std::stod(data[4]) / CM_TO_M;

		double gamma_self = std::stod(data[5]) / CM_TO_M / ATM_TO_PA;
		double gamma_air = std::stod(data[6]) / CM_TO_M / ATM_TO_PA;
		double n_air = std::stod(data[7]);
		double delta_air = std::stod(data[8]) / CM_TO_M / ATM_TO_PA;
		double gamma_CO2 = std::stod(data[9]) / CM_TO_M / ATM_TO_PA;

		double n_CO2 = std::stod(data[10]);
		double delta_CO2 = std::stod(data[11]) / CM_TO_M / ATM_TO_PA;
		double gamma_H2 = std::stod(data[12]) / CM_TO_M / ATM_TO_PA;
		double n_H2 = std::stod(data[13]);
		double delta_H2 = std::stod(data[14]) / CM_TO_M / ATM_TO_PA;

		double gamma_He = std::stod(data[15]) / CM_TO_M / ATM_TO_PA;
		double n_He = std::stod(data[16]);
		double delta_He = std::stod(data[17]) / CM_TO_M / ATM_TO_PA;
		double gamma_H2O = std::stod(data[18]) / CM_TO_M / ATM_TO_PA;
		double n_H2O = std::stod(data[19]);

		Line L;
		L.molecular_id = isotopologue.molecule_id;
		L.local_id = isotopologue.local_iso_id;
		L.global_id = isotopologue.global_id;

		L.nu = nu;
		L.sw = sw;
		L.elower = elower;
		L.gamma_self = gamma_self;
		L.gamma_air = gamma_air;
		L.n_air = n_air;
		L.delta_air = delta_air;
		
		L.gamma_CO2 = gamma_CO2;
		L.n_CO2 = n_CO2;
		L.delta_CO2 = delta_CO2;

		L.gamma_H2 = gamma_H2;
		L.n_H2 = n_H2;
		L.delta_H2 = delta_H2;

		L.gamma_He = gamma_He;
		L.n_He = n_He;
		L.delta_He = delta_He;

		L.gamma_H2O = gamma_H2O;
		L.n_H2O = n_H2O;
		L.delta_H2O = std::numeric_limits<double>::quiet_NaN();

		out.push_back(L);
	}

	return out;
}

inline std::vector<Line> loadLines_raw(const std::string& filename, int molecule_id, std::vector<int> local_ids, double nu_min, double nu_max, bool header = true)
{
	if (nu_max < nu_min)
	{
		std::swap(nu_min, nu_max);
	}

	std::ifstream fin(filename);
	if (!fin.is_open())
	{
		throw std::runtime_error("Cannot open line file: " + filename);
	}

	std::vector<Line> out;
	std::string line;

	if(header == true)
	{
		std::getline(fin, line);
	}

	while(std::getline(fin, line))
	{
		// std::cout << line << std::endl;
		auto data = splitString(line, ',');

		if (data.size() != 54)
		{
			throw std::runtime_error("Invalid format: " + filename);
		}

		for(int i = 0; i < data.size(); i ++)
		{
			if(data[i] == "#")
			{
				// data[i] = "0.0";
				data[i] = std::to_string(std::numeric_limits<double>::quiet_NaN());
			}
		}

		int molec_id = std::stoi(data[0]);
		int local_iso_id = std::stoi(data[1]);
		if (molec_id != molecule_id || !std::any_of(local_ids.begin(), local_ids.end(), [&](int x){ return x == local_iso_id;}))
		{
			continue;
		}

		double nu = std::stod(data[2]) / CM_TO_M;
		if (nu < nu_min || nu_max < nu)
		{
			continue;
		}

		double sw = std::stod(data[3]) * CM_TO_M;
		double elower = std::stod(data[4]) / CM_TO_M;

		double gamma_self = std::stod(data[5]) / CM_TO_M / ATM_TO_PA;
		double gamma_air = std::stod(data[6]) / CM_TO_M / ATM_TO_PA;
		double n_air = std::stod(data[7]);
		double delta_air = std::stod(data[8]) / CM_TO_M / ATM_TO_PA;
		double gamma_CO2 = std::stod(data[9]) / CM_TO_M / ATM_TO_PA;

		double n_CO2 = std::stod(data[10]);
		double delta_CO2 = std::stod(data[11]) / CM_TO_M / ATM_TO_PA;
		double gamma_H2 = std::stod(data[12]) / CM_TO_M / ATM_TO_PA;
		double n_H2 = std::stod(data[13]);
		double delta_H2 = std::stod(data[14]) / CM_TO_M / ATM_TO_PA;

		double gamma_He = std::stod(data[15]) / CM_TO_M / ATM_TO_PA;
		double n_He = std::stod(data[16]);
		double delta_He = std::stod(data[17]) / CM_TO_M / ATM_TO_PA;
		double gamma_H2O = std::stod(data[18]) / CM_TO_M / ATM_TO_PA;
		double n_H2O = std::stod(data[19]);

		Line L;
		L.molecular_id = molecule_id;
		L.local_id = local_iso_id;
		L.global_id = global_from_mol_local(molec_id, local_iso_id);

		L.nu = nu;
		L.sw = sw;
		L.elower = elower;
		L.gamma_self = gamma_self;
		L.gamma_air = gamma_air;
		L.n_air = n_air;
		L.delta_air = delta_air;
		
		L.gamma_CO2 = gamma_CO2;
		L.n_CO2 = n_CO2;
		L.delta_CO2 = delta_CO2;

		L.gamma_H2 = gamma_H2;
		L.n_H2 = n_H2;
		L.delta_H2 = delta_H2;

		L.gamma_He = gamma_He;
		L.n_He = n_He;
		L.delta_He = delta_He;

		L.gamma_H2O = gamma_H2O;
		L.n_H2O = n_H2O;

		out.push_back(L);
	}

	return out;
}

inline std::vector<std::string> readCIAHeader_raw(std::string header)
{
	if(header.size() < 100)
	{
		throw std::runtime_error("Not a header line: " + header);
	}

	std::string chemical_symbol = header.substr(0, 20);

	std::string mole1, mole2;

	for(int i = 0; i < chemical_symbol.size(); i ++)
	{
		if(chemical_symbol[i] == ' ')
		{
			chemical_symbol.erase(i, 1);
			i --;
		}
	}

	auto moles = splitString(chemical_symbol, '-');
	mole1 = moles[0];
	mole2 = moles[1];
	
	std::string wn_min = header.substr(20, 10);
	for(int i = 0; i < wn_min.size(); i ++)
	{
		if(wn_min[i] == ' ')
		{
			wn_min.erase(i, 1);
			i --;
		}
	}

	std::string wn_max = header.substr(30, 10);
	for(int i = 0; i < wn_max.size(); i ++)
	{
		if(wn_max[i] == ' ')
		{
			wn_max.erase(i, 1);
			i --;
		}
	}

	std::string number_pts = header.substr(40, 7);
	for(int i = 0; i < number_pts.size(); i ++)
	{
		if(number_pts[i] == ' ')
		{
			number_pts.erase(i, 1);
			i --;
		}
	}

	std::string temperature = header.substr(47, 7);
	for(int i = 0; i < temperature.size(); i ++)
	{
		if(temperature[i] == ' ')
		{
			temperature.erase(i, 1);
			i --;
		}
	}

	std::string maximum_cia = header.substr(54, 10);
	for(int i = 0; i < maximum_cia.size(); i ++)
	{
		if(maximum_cia[i] == ' ')
		{
			maximum_cia.erase(i, 1);
			i --;
		}
	}

	std::string resolution = header.substr(64, 6);
	for(int i = 0; i < resolution.size(); i ++)
	{
		if(resolution[i] == ' ')
		{
			resolution.erase(i, 1);
			i --;
		}
	}

	std::string comments = header.substr(70, 27);
	for(int i = 0; i < comments.size(); i ++)
	{
		if(comments[i] == ' ')
		{
			comments.erase(i, 1);
			i --;
		}
	}

	std::string ref_no = header.substr(97, 3);
	for(int i = 0; i < ref_no.size(); i ++)
	{
		if(ref_no[i] == ' ')
		{
			ref_no.erase(i, 1);
			i --;
		}
	}

	std::vector<std::string> result = {mole1, mole2, wn_min, wn_max, number_pts, temperature, maximum_cia, resolution, comments, ref_no};

	return result;
}

inline void generateNetCDF(const std::string& filename, const std::string& input_directory)
{
	std::string dir = input_directory;

	if(dir[dir.size() - 1] != '/')
	{
		dir = dir + "/";
	}

	std::string dir_line_by_line = dir + "LBL/";
	std::string dir_cia = dir + "CIA/";
	std::string dir_qtable = dir + "Q_tables/";

	std::cout << "Starting NetCDF generation: " << filename << std::endl;
	std::cout << "--------------------------------------------------" << std::endl;

	netCDF::NcFile outputFile(filename, netCDF::NcFile::replace);

	netCDF::NcGroup group_lbl = outputFile.addGroup("line_by_line");

	for(int m = 0; m < molecules.size(); m ++)
	{
		std::cout << "[" << std::setw(2) << m + 1 << "/" << molecules.size() << "] " << "Processing Molecule: " << std::left << std::setw(8) << molecules[m].name << " (ID: " << molecules[m].molecule_id << ")" << std::endl;
				
		netCDF::NcGroup group_mol = group_lbl.addGroup(std::to_string(molecules[m].molecule_id));

		group_mol.putAtt("name", molecules[m].name);
		group_mol.putAtt("formula", molecules[m].general_formula);
		group_mol.putAtt("id", netCDF::ncInt, molecules[m].molecule_id);
		
		for(int i = 0; i < isotopologues.size(); i ++)
		{
			if(isotopologues[i].molecule_id == molecules[m].molecule_id)
			{
				auto iso = iso_from_global(isotopologues[i].global_id);

				std::cout << "  -> Isotopologue: " << iso.isotopic_formula << std::flush;

				netCDF::NcGroup group_iso = group_mol.addGroup(std::to_string(iso.global_id));
				group_iso.putAtt("formula", iso.isotopic_formula);

				group_iso.putAtt("id", netCDF::ncInt, iso.global_id);
				group_iso.putAtt("local_id", netCDF::ncInt, iso.local_iso_id);
				group_iso.putAtt("afgl_code", netCDF::ncInt, iso.afgl_code);
				group_iso.putAtt("abundance_earth", netCDF::ncDouble, iso.abundance);
				group_iso.putAtt("molar_mass", netCDF::ncDouble, iso.molar_mass);
				group_iso.putAtt("nuclear_spin_weight", netCDF::ncInt, iso.gi);

				try
				{
					auto lines = loadLines_raw(dir_line_by_line + molecules[m].general_formula + ".txt", iso, 0.0, 1.0E100);
					
					netCDF::NcGroup group_line = group_iso.addGroup("lines");
					std::vector<double> nu, sw, elower, gamma_self;
					std::vector<double> gamma_air, n_air, delta_air;
					std::vector<double> gamma_CO2, n_CO2, delta_CO2;
					std::vector<double> gamma_H2, n_H2, delta_H2;
					std::vector<double> gamma_He, n_He, delta_He;
					std::vector<double> gamma_H2O, n_H2O, delta_H2O;

					for(int j = 0; j < lines.size(); j ++)
					{
						nu.push_back(lines[j].nu);
						sw.push_back(lines[j].sw);
						elower.push_back(lines[j].elower);
						gamma_self.push_back(lines[j].gamma_self);

						gamma_air.push_back(lines[j].gamma_air);
						n_air.push_back(lines[j].n_air);
						delta_air.push_back(lines[j].delta_air);

						gamma_CO2.push_back(lines[j].gamma_CO2);
						n_CO2.push_back(lines[j].n_CO2);
						delta_CO2.push_back(lines[j].delta_CO2);

						gamma_H2.push_back(lines[j].gamma_H2);
						n_H2.push_back(lines[j].n_H2);
						delta_H2.push_back(lines[j].delta_H2);

						gamma_He.push_back(lines[j].gamma_He);
						n_He.push_back(lines[j].n_He);
						delta_He.push_back(lines[j].delta_He);

						gamma_H2O.push_back(lines[j].gamma_H2O);
						n_H2O.push_back(lines[j].n_H2O);
						delta_H2O.push_back(lines[j].delta_H2O);
					}
			
					netCDF::NcDim dim_lines = group_line.addDim("line", lines.size());

					netCDF::NcVar var_nu = group_line.addVar("wavenumber", netCDF::ncDouble, dim_lines);
					var_nu.putAtt("description", "the wavenumber of the spectral line transition in vacuum");
					var_nu.putAtt("units", "m-1");
					var_nu.putVar(nu.data());
					netCDF::NcVar var_sw = group_line.addVar("S", netCDF::ncDouble, dim_lines);
					var_sw.putAtt("description", "line intensity");
					var_sw.putAtt("units", "m/molecule");
					var_sw.putVar(sw.data());
					netCDF::NcVar var_elower = group_line.addVar("elower", netCDF::ncDouble, dim_lines);
					var_elower.putAtt("description", "lower-state energy of the transition");
					var_elower.putAtt("units", "m-1");
					var_elower.putVar(elower.data());
					netCDF::NcVar var_gamma_self = group_line.addVar("gamma_self", netCDF::ncDouble, dim_lines);
					var_gamma_self.putAtt("description", "lorentzian half-width at half-maximum (HWHW) due to self-broadening (collisions with the same species), at reference terperature and pressure");
					var_gamma_self.putAtt("units", "m-1/Pa");
					var_gamma_self.putVar(gamma_self.data());

					netCDF::NcVar var_gamma_air = group_line.addVar("gamma_air", netCDF::ncDouble, dim_lines);
					var_gamma_air.putAtt("description", "lorentzian (HWHW) due to air-broadening");
					var_gamma_air.putAtt("units", "m-1/Pa");
					var_gamma_air.putVar(gamma_air.data());
					netCDF::NcVar var_n_air = group_line.addVar("n_air", netCDF::ncDouble, dim_lines);
					var_n_air.putAtt("description", "temperature exponent for the air-broadened half-width");
					var_n_air.putAtt("units", "dimensionless");
					var_n_air.putVar(n_air.data());
					netCDF::NcVar var_delta_air = group_line.addVar("delta_air", netCDF::ncDouble, dim_lines);
					var_delta_air.putAtt("description", "pressure shift of the line center due to collisions with air molecules");
					var_delta_air.putAtt("units", "m-1/Pa");
					var_delta_air.putVar(delta_air.data());

					netCDF::NcVar var_gamma_CO2 = group_line.addVar("gamma_CO2", netCDF::ncDouble, dim_lines);
					var_gamma_CO2.putAtt("description", "lorentzian (HWHW) due to air-broadening");
					var_gamma_CO2.putAtt("units", "m-1/Pa");
					var_gamma_CO2.putVar(gamma_CO2.data());
					netCDF::NcVar var_n_CO2 = group_line.addVar("n_CO2", netCDF::ncDouble, dim_lines);
					var_n_CO2.putAtt("description", "temperature exponent for the air-broadened half-width");
					var_n_CO2.putAtt("units", "dimensionless");
					var_n_CO2.putVar(n_CO2.data());
					netCDF::NcVar var_delta_CO2 = group_line.addVar("delta_CO2", netCDF::ncDouble, dim_lines);
					var_delta_CO2.putAtt("description", "pressure shift of the line center due to collisions with air molecules");
					var_delta_CO2.putAtt("units", "m-1/Pa");
					var_delta_CO2.putVar(delta_CO2.data());

					netCDF::NcVar var_gamma_H2 = group_line.addVar("gamma_H2", netCDF::ncDouble, dim_lines);
					var_gamma_H2.putAtt("description", "lorentzian (HWHW) due to air-broadening");
					var_gamma_H2.putAtt("units", "m-1/Pa");
					var_gamma_H2.putVar(gamma_H2.data());
					netCDF::NcVar var_n_H2 = group_line.addVar("n_H2", netCDF::ncDouble, dim_lines);
					var_n_H2.putAtt("description", "temperature exponent for the air-broadened half-width");
					var_n_H2.putAtt("units", "dimensionless");
					var_n_H2.putVar(n_H2.data());
					netCDF::NcVar var_delta_H2 = group_line.addVar("delta_H2", netCDF::ncDouble, dim_lines);
					var_delta_H2.putAtt("description", "pressure shift of the line center due to collisions with air molecules");
					var_delta_H2.putAtt("units", "m-1/Pa");
					var_delta_H2.putVar(delta_H2.data());

					netCDF::NcVar var_gamma_He = group_line.addVar("gamma_He", netCDF::ncDouble, dim_lines);
					var_gamma_He.putAtt("description", "lorentzian (HWHW) due to air-broadening");
					var_gamma_He.putAtt("units", "m-1/Pa");
					var_gamma_He.putVar(gamma_He.data());
					netCDF::NcVar var_n_He = group_line.addVar("n_He", netCDF::ncDouble, dim_lines);
					var_n_He.putAtt("description", "temperature exponent for the air-broadened half-width");
					var_n_He.putAtt("units", "dimensionless");
					var_n_He.putVar(n_He.data());
					netCDF::NcVar var_delta_He = group_line.addVar("delta_He", netCDF::ncDouble, dim_lines);
					var_delta_He.putAtt("description", "pressure shift of the line center due to collisions with air molecules");
					var_delta_He.putAtt("units", "m-1/Pa");
					var_delta_He.putVar(delta_He.data());

					netCDF::NcVar var_gamma_H2O = group_line.addVar("gamma_H2O", netCDF::ncDouble, dim_lines);
					var_gamma_H2O.putAtt("description", "lorentzian (HWHW) due to air-broadening");
					var_gamma_H2O.putAtt("units", "m-1/Pa");
					var_gamma_H2O.putVar(gamma_H2O.data());
					netCDF::NcVar var_n_H2O = group_line.addVar("n_H2O", netCDF::ncDouble, dim_lines);
					var_n_H2O.putAtt("description", "temperature exponent for the air-broadened half-width");
					var_n_H2O.putAtt("units", "dimensionless");
					var_n_H2O.putVar(n_H2O.data());
					netCDF::NcVar var_delta_H2O = group_line.addVar("delta_H2O", netCDF::ncDouble, dim_lines);
					var_delta_H2O.putAtt("description", "pressure shift of the line center due to collisions with air molecules");
					var_delta_H2O.putAtt("units", "m-1/Pa");
					var_delta_H2O.putVar(delta_H2O.data());

					std::cout << " (" << lines.size() << " lines)" << std::flush;
				}
				catch(const std::exception& e)
				{
					std::cerr << "Skipping isotopologue " << iso.isotopic_formula << " (" << e.what() << ")" << std::endl;
				}

				std::cout << " + Q-table" << std::endl;
				try
				{
					iso.loadQTable_raw(dir_qtable + "q" + std::to_string(iso.global_id) + ".txt");

					netCDF::NcGroup group_q = group_iso.addGroup("partition_function");
					netCDF::NcDim dim_temperature = group_q.addDim("temperature", iso.qtable.T.size());

					netCDF::NcVar var_T = group_q.addVar("T", netCDF::ncDouble, dim_temperature);
					var_T.putAtt("description", "temperature");
					var_T.putAtt("units", "K");
					var_T.putVar(iso.qtable.T.data());

					netCDF::NcVar var_Q = group_q.addVar("Q", netCDF::ncDouble, dim_temperature);
					var_Q.putAtt("description", "partition function");
					var_Q.putAtt("units", "dimensionless");
					var_Q.putVar(iso.qtable.Q.data());
				}
				catch(const std::exception& e)
				{
					std::cerr << "Skipping isotopologue " << iso.isotopic_formula << " (" << e.what() << ")" << std::endl;
				}
			}
		}
	}

	auto cia_list = listFiles(dir_cia, "cia", true, true);

	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "Processing CIA data (Total " << cia_list.size() << " files)..." << std::endl;

	netCDF::NcGroup group_cia = outputFile.addGroup("collision_induced_absorption");

	for(int i = 0; i < cia_list.size(); i ++)
	{
		CIAGrid grid;
		grid.loadFromASCII(cia_list[i]);

		std::cout << "[" << i + 1 << "/" << cia_list.size() << "] CIA Pair: " << grid.molecule1 << "-" << grid.molecule2 << std::endl;

		std::string group_name = grid.molecule1 + "_" + grid.molecule2;
		if(grid.ortho_para_ratio != "n/a")
		{
			group_name += "_" + grid.ortho_para_ratio;
		}

		netCDF::NcGroup group_mol = group_cia.addGroup(group_name);
		group_mol.putAtt("molecule_1", grid.molecule1);
		group_mol.putAtt("molecule_2", grid.molecule2);

		if(grid.ortho_para_ratio == "normal")
		{
			group_mol.putAtt("ortho_para_ratio", "3:1");
		}
		else if(grid.ortho_para_ratio != "n/a")
		{
			group_mol.putAtt("ortho_para_ratio", grid.ortho_para_ratio);
		}

		for(size_t j = 0; j < grid.temperatures.size(); j ++)
		{
			netCDF::NcGroup group_temp = group_mol.addGroup(std::to_string(j));
			group_temp.putAtt("temperature", netCDF::ncDouble, grid.temperatures[j]);

			netCDF::NcDim dim_wavenumber = group_temp.addDim("wavenumber", grid.wavenumbers[j].size());
			netCDF::NcVar var_wavenumber = group_temp.addVar("wavenumber", netCDF::ncDouble, dim_wavenumber);
			var_wavenumber.putAtt("description", "wavenumber");
			var_wavenumber.putAtt("units", "m-1");
			var_wavenumber.putVar(grid.wavenumbers[j].data());

			netCDF::NcVar var_cia = group_temp.addVar("k", netCDF::ncDouble, dim_wavenumber);
			var_cia.putAtt("description", "collision-induced absorption coefficient");
			var_cia.putAtt("units", "m5/molecule2");
			var_cia.putVar(grid.k_cia[j].data());
		}
	}

	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "Successfully generated: " << filename << std::endl;

	return;
}

inline void CIAGrid::loadFromASCII(const std::string& filename)
{
	temperatures.clear();
	wavenumbers.clear();
	k_cia.clear();

	std::ifstream input_cia(filename);
	std::string str;

	while(std::getline(input_cia, str))
	{
		try
		{
			auto header = readCIAHeader_raw(str);

			if (temperatures.empty())
			{
				molecule1 = header[0];
				molecule2 = header[1];
				
				if(header[8] == "Normal")
				{
					ortho_para_ratio = "normal";
				}
				else if(header[8] == "Equilibrium")
				{
					ortho_para_ratio = "equilibrium";
				}
				else
				{
					ortho_para_ratio = "n/a";
				}
			}

			double current_T = std::stod(header[5]);
			int n_pts = std::stoi(header[4]);

			std::vector<double> current_wn;
			std::vector<double> current_k;
		
			for(int i = 0; i < n_pts; i ++)
			{
				std::getline(input_cia, str);
				double wn, sigma;

				if(std::sscanf(str.data(), "%lf %lf", &wn, &sigma) == 2)
				{
					current_wn.push_back(wn / CM_TO_M);
					current_k.push_back(sigma * 1.0E-10);
				}
				else
				{
					throw std::runtime_error("Format error: " + str);
				}
			}

			temperatures.push_back(current_T);
			wavenumbers.push_back(current_wn);
			k_cia.push_back(current_k);
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}

inline void CIAGrid::loadCIA(const netCDF::NcGroup& cia_group)
{
	int t_idx = 0;

	struct TempData
	{
		double T;
		std::vector<double> nu;
		std::vector<double> k;
	};

	std::vector<TempData> raw_data;

	while (true)
	{
		netCDF::NcGroup t_group = cia_group.getGroup(std::to_string(t_idx));
		if (t_group.isNull()) break;

		TempData td;
		netCDF::NcGroupAtt att_t = t_group.getAtt("temperature");
		
		if (att_t.getType() == netCDF::ncDouble)
		{
			att_t.getValues(&td.T);
		}
		else
		{
			std::string t_str;
			att_t.getValues(t_str);
			td.T = std::stod(t_str);
		}

		netCDF::NcDim dim_nu = t_group.getDim("wavenumber");
		int n_pts = dim_nu.getSize();

		td.nu.resize(n_pts);
		td.k.resize(n_pts);

		t_group.getVar("wavenumber").getVar(td.nu.data());
		t_group.getVar("k").getVar(td.k.data());

		raw_data.push_back(td);
		t_idx++;
	}

	std::sort(raw_data.begin(), raw_data.end(), [](const TempData& a, const TempData& b){return a.T < b.T;});

	for (const auto& td : raw_data)
	{
		temperatures.push_back(td.T);
		wavenumbers.push_back(td.nu);
		k_cia.push_back(td.k);
	}
}

template <typename T> inline T CIAGrid::evaluate(double nu, T target_T) const
{
	if (temperatures.empty()) return T(0.0);

	std::vector<size_t> valid_indices;
	for (size_t i = 0; i < temperatures.size(); ++i)
	{
		if (nu >= wavenumbers[i].front() && nu <= wavenumbers[i].back())
		{
			valid_indices.push_back(i);
		}
	}

	if (valid_indices.empty())
	{
		return T(0.0);
	}

	double t_val = autodiff::get_value(target_T);

	if (t_val <= temperatures[valid_indices.front()])
	{
		return T(interpolate1D(nu, wavenumbers[valid_indices.front()], k_cia[valid_indices.front()]));
	}

	if (t_val >= temperatures[valid_indices.back()])
	{
		return T(interpolate1D(nu, wavenumbers[valid_indices.back()], k_cia[valid_indices.back()]));
	}

	auto it = std::lower_bound(valid_indices.begin(), valid_indices.end(), t_val, [&](size_t idx, double val) {return temperatures[idx] < val;});

	size_t idx_high = *it;
	size_t idx_low = *(--it);

	double t_low = temperatures[idx_low];
	double t_high = temperatures[idx_high];
	double k_low = interpolate1D(nu, wavenumbers[idx_low], k_cia[idx_low]);
	double k_high = interpolate1D(nu, wavenumbers[idx_high], k_cia[idx_high]);

	if (k_low <= 1.0e-30 || k_high <= 1.0e-30)
	{
		T ratio = (target_T - T(t_low)) / T(t_high - t_low);
		return T(k_low) + ratio * T(k_high - k_low);
	}

	using std::log; using autodiff::log;
	using std::exp; using autodiff::exp;

	T inv_T_target = T(1.0) / target_T;
	T ratio = (inv_T_target - T(1.0/t_low)) / T(1.0/t_high - 1.0/t_low);
	T log_k_target = T(std::log(k_low)) + ratio * T(std::log(k_high) - std::log(k_low));

	return exp(log_k_target);
}

inline double CIAGrid::interpolate1D(double target_nu, const std::vector<double>& nu_grid, const std::vector<double>& k_grid) const
{
	if (nu_grid.empty() || target_nu < nu_grid.front() || target_nu > nu_grid.back())
	{
		return 0.0;
	}
	
	auto it = std::lower_bound(nu_grid.begin(), nu_grid.end(), target_nu);
	int idx_high = std::distance(nu_grid.begin(), it);

	if (idx_high == 0)
	{
		return k_grid.front();
	}

	int idx_low = idx_high - 1;
	double nu_low = nu_grid[idx_low];
	double nu_high = nu_grid[idx_high];
	
	if (nu_high == nu_low)
	{
		return k_grid[idx_low];
	}

	double ratio = (target_nu - nu_low) / (nu_high - nu_low);
	return k_grid[idx_low] + ratio * (k_grid[idx_high] - k_grid[idx_low]);
}

inline const std::vector<CIAGrid>& CIA::getLoadedGrids(void) const
{
	return loaded_grids;
}

inline bool CIA::hasGroup(const netCDF::NcFile& nc, const std::string& group_name) const
{
	try
	{
		return !nc.getGroup("collision_induced_absorption").getGroup(group_name).isNull();
	}
	catch(...)
	{
		return false;
	}
}

inline void CIA::load(const netCDF::NcFile& nc, const std::vector<std::string>& molecules)
{
	loaded_grids.clear();
	netCDF::NcGroup base_group = nc.getGroup("collision_induced_absorption");

	if (base_group.isNull())
	{
		return;
	}

	std::multimap<std::string, netCDF::NcGroup> pairs = base_group.getGroups();

	for (auto const& [group_name, group] : pairs)
	{
		std::string mol1, mol2;

		try
		{
			netCDF::NcGroupAtt att1 = group.getAtt("molecule_1");
			netCDF::NcGroupAtt att2 = group.getAtt("molecule_2");

			if (att1.isNull() || att2.isNull())
			{
				continue;
			}

			att1.getValues(mol1);
			att2.getValues(mol2);
		}
		catch (...)
		{
			continue;
		}

		auto it1 = std::find(molecules.begin(), molecules.end(), mol1);
		auto it2 = std::find(molecules.begin(), molecules.end(), mol2);

		if (it1 != molecules.end() && it2 != molecules.end())
		{
			CIAGrid grid;
			grid.molecule1 = mol1;
			grid.molecule2 = mol2;
			
			netCDF::NcGroupAtt att_opr = group.getAtt("ortho_para_ratio");
			if (!att_opr.isNull())
			{
				att_opr.getValues(grid.ortho_para_ratio);
			}

			grid.loadCIA(group);
			loaded_grids.push_back(grid);
		}
	}
}

template <typename T> inline T CIA::computeAbsorptionCoefficient(double nu, T target_T, const std::map<std::string, T>& number_densities) const
{
	T total_alpha = T(0.0);

	for (const auto& grid : loaded_grids)
	{
		auto it1 = number_densities.find(grid.molecule1);
		auto it2 = number_densities.find(grid.molecule2);

		if (it1 != number_densities.end() && it2 != number_densities.end())
		{
			T k_cia = grid.evaluate(nu, target_T); 
			total_alpha += k_cia * it1->second * it2->second;
		}
	}

	return total_alpha;
}

}
