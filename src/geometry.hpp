#pragma once

#include <cmath>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include <Eigen/Dense>

namespace paad::geometry
{

struct Geometry
{
	int M;
	int Ntheta;
	int Nphi;

	double d_phi;
	
	Eigen::VectorXd theta_uh;
	Eigen::VectorXd theta_lh;
	Eigen::VectorXd theta_all;

	Eigen::VectorXd phi;

	Eigen::VectorXd weight;
	Eigen::VectorXd mu;
	Eigen::MatrixXd WMU;
};

std::vector<Eigen::MatrixXd> reconstruct(const std::vector<Eigen::MatrixXd>& F_mode, const Geometry& geo);
std::vector<Eigen::MatrixXd> reconstruct(const std::vector<Eigen::MatrixXd>& F_mode, const std::vector<double>& out_phi, int n_theta);
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> computeFourierSeriesCoefficients(const std::vector<Eigen::MatrixXd>& f_phi, const Geometry& geo);
std::vector<Eigen::MatrixXd> computePackedFourierCoefficients(const std::vector<Eigen::MatrixXd>& f_phi, const Geometry& geo);

Geometry generateGeometryGaussRadau(int n_theta, int n_phi = -1, int n_mode = -1);
Geometry generateGeometryRegular(int n_theta, int n_phi = -1, int n_mode = -1, double eps = 1.0E-3);


inline Eigen::Matrix4d getRotationMatrix(double cos_2beta, double sin_2beta)
{
	Eigen::Matrix4d L = Eigen::Matrix4d::Identity();
	L(1, 1) = cos_2beta;
	L(1, 2) = sin_2beta;
	L(2, 1) = -sin_2beta;
	L(2, 2) = cos_2beta;
	return L;
}

inline Eigen::Matrix4d rotateMuellerMatrix(const Eigen::Matrix4d& F, double alpha1, double alpha2)
{
	auto L = [](double chi)
	{
		double c2 = std::cos(2.0 * chi);
		double s2 = std::sin(2.0 * chi);
		Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
		m(1, 1) =  c2; m(1, 2) = s2;
		m(2, 1) = -s2; m(2, 2) = c2;
		return m;
	};
	
	return L(-alpha2) * F * L(-alpha1);
}

inline Eigen::Matrix4d computeZMatrix(double mu, double mu_prime, double d_phi, const std::function<Eigen::Matrix4d(double)>& phase_func)
{
	double su = std::sqrt(std::max(0.0, 1.0 - mu * mu));
	double su_prime = std::sqrt(std::max(0.0, 1.0 - mu_prime * mu_prime));
	double cp = std::cos(d_phi);
	double sp = std::sin(d_phi);

	double cos_Theta = mu * mu_prime + su * su_prime * cp;
	cos_Theta = std::clamp(cos_Theta, -1.0, 1.0);
	double sin_Theta = std::sqrt(1.0 - cos_Theta * cos_Theta);

	if (sin_Theta < 1e-12)
	{
		return phase_func(cos_Theta);
	}

	double cos_s1 = (-mu + mu_prime * cos_Theta) / (su_prime * sin_Theta);
	double cos_s2 = (-mu_prime + mu * cos_Theta) / (su * sin_Theta);

	cos_s1 = std::clamp(cos_s1, -1.0, 1.0);
	cos_s2 = std::clamp(cos_s2, -1.0, 1.0);

	double s1 = std::acos(cos_s1);
	double s2 = std::acos(cos_s2);

	if (-sp < 0) 
	{
		s1 = -s1;
		s2 = -s2;
	}

	double cos_2s1 = std::cos(2.0 * s1);
	double sin_2s1 = std::sin(2.0 * s1);
	double cos_2s2 = std::cos(2.0 * s2);
	double sin_2s2 = std::sin(2.0 * s2);

	Eigen::Matrix4d L2 = getRotationMatrix(cos_2s2, -sin_2s2); 
	Eigen::Matrix4d L1 = getRotationMatrix(cos_2s1, -sin_2s1);  
	
	return L2 * phase_func(cos_Theta) * L1;
}

inline void computeScatteringGeometry(const Geometry& geo, double mu_e, double mu_i, double phi_diff, double& theta, double& rot1, double& rot2)
{
	using std::sqrt; using std::acos; using std::sin; using std::cos;
	
	double su_prime = std::sqrt(std::max(0.0, 1.0 - mu_i * mu_i));
	double su = std::sqrt(std::max(0.0, 1.0 - mu_e * mu_e));
	double cp = std::cos(phi_diff);
	double sp = std::sin(phi_diff);

	double cos_theta = mu_e * mu_i + su * su_prime * cp;
	cos_theta = std::clamp(cos_theta, -1.0, 1.0);
	theta = std::acos(cos_theta);
	
	double st = std::sin(theta);

	if (st < 1.0E-12 || su_prime < 1.0E-12 || su < 1.0E-12)
	{
		rot1 = 0.0;
		rot2 = 0.0;
		return;
	}

	double cos_a1 = (-mu_e + mu_i * cos_theta) / (su_prime * st);
	double cos_a2 = (-mu_i + mu_e * cos_theta) / (su * st);
	
	cos_a1 = std::clamp(cos_a1, -1.0, 1.0);
	cos_a2 = std::clamp(cos_a2, -1.0, 1.0);
	
	rot1 = std::acos(cos_a1);
	rot2 = std::acos(cos_a2);

	if (-sp < 0) 
	{
		rot1 = -rot1;
		rot2 = -rot2;
	}
}

}
