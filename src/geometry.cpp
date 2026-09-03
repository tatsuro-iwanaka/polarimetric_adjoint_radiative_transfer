#include "geometry.hpp"
#include <numbers>
#include "math.hpp"

namespace paad::geometry
{

std::vector<Eigen::MatrixXd> reconstruct(const std::vector<Eigen::MatrixXd>& F_mode, const Geometry& geo)
{
	int dim = F_mode[0].rows();
	int n_stokes = dim / geo.Ntheta;
	
	std::vector<Eigen::MatrixXd> f_phi(geo.Nphi, Eigen::MatrixXd::Zero(dim, dim));

	for(int l = 0; l < geo.Nphi; l++)
	{
		f_phi[l] += F_mode[0];

		for(int m = 1; m <= geo.M; m++)
		{
			double cos_ml = std::cos(double(m) * geo.phi[l]);
			double sin_ml = std::sin(double(m) * geo.phi[l]);
			
			Eigen::MatrixXd m_eff = Eigen::MatrixXd::Zero(dim, dim);

			if (n_stokes == 1)
			{
				m_eff = F_mode[m] * cos_ml;
			}
			else if (n_stokes == 3)
			{
				for (int i = 0; i < geo.Ntheta; ++i)
				{
					for (int j = 0; j < geo.Ntheta; ++j)
					{
						int r = 3 * i; int c = 3 * j;

						m_eff.block<2, 2>(r, c) = F_mode[m].block<2, 2>(r, c) * cos_ml;
						m_eff(r+2, c+2) = F_mode[m](r+2, c+2) * cos_ml;
						
						m_eff.block<2, 1>(r, c+2) = -F_mode[m].block<2, 1>(r, c+2) * sin_ml;
						m_eff.block<1, 2>(r+2, c) = F_mode[m].block<1, 2>(r+2, c) * sin_ml;
					}
				}
			}
			else if (n_stokes == 4)
			{
				for (int i = 0; i < geo.Ntheta; ++i)
				{
					for (int j = 0; j < geo.Ntheta; ++j)
					{
						int r = 4 * i; int c = 4 * j;

						m_eff.block<2, 2>(r, c) = F_mode[m].block<2, 2>(r, c) * cos_ml;
						m_eff.block<2, 2>(r+2, c+2) = F_mode[m].block<2, 2>(r+2, c+2) * cos_ml;
						
						m_eff.block<2, 2>(r, c+2) = -F_mode[m].block<2, 2>(r, c+2) * sin_ml;
						m_eff.block<2, 2>(r+2, c) = F_mode[m].block<2, 2>(r+2, c) * sin_ml;
					}
				}
			}

			f_phi[l] += 2.0 * m_eff;
		}
	}
	return f_phi;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> computeFourierSeriesCoefficients(const std::vector<Eigen::MatrixXd>& f_phi, const Geometry& geo)
{
	int dim = f_phi[0].rows();
	double factor = geo.d_phi / (2.0 * std::numbers::pi);

	std::vector<Eigen::MatrixXd> Fcos(geo.M + 1, Eigen::MatrixXd::Zero(dim, dim));
	std::vector<Eigen::MatrixXd> Fsin(geo.M + 1, Eigen::MatrixXd::Zero(dim, dim));

	Eigen::MatrixXd C = Eigen::MatrixXd::Ones((geo.M + 1), geo.Nphi);
	Eigen::MatrixXd S = Eigen::MatrixXd::Zero((geo.M + 1), geo.Nphi);

	Eigen::RowVectorXd cphi(geo.Nphi);
	Eigen::RowVectorXd sphi(geo.Nphi);

	for(int p = 0; p < geo.Nphi; p++)
	{
		cphi(p) = std::cos(geo.phi[p]);
		sphi(p) = std::sin(geo.phi[p]);
	}

	for(int i = 1; i <= geo.M; i++)
	{
		C.row(i) = C.row(i - 1).cwiseProduct(cphi) - S.row(i - 1).cwiseProduct(sphi);
		S.row(i) = S.row(i - 1).cwiseProduct(cphi) + C.row(i - 1).cwiseProduct(sphi);
	}

	for(int i = 0; i <= geo.M; i++)
	{
		for(int p = 0; p < geo.Nphi; p++)
		{
			Fcos[i] += f_phi[p] * (C(i, p) * factor);
			Fsin[i] += f_phi[p] * (S(i, p) * factor);
		}
	}
	
	return {Fcos, Fsin};
}

std::vector<Eigen::MatrixXd> computePackedFourierCoefficients(const std::vector<Eigen::MatrixXd>& f_phi, const Geometry& geo)
{
	auto [Fcos, Fsin] = computeFourierSeriesCoefficients(f_phi, geo);

	int dim = Fcos[0].rows();
	int n_stokes = dim / geo.Ntheta;
	std::vector<Eigen::MatrixXd> F_packed(geo.M + 1, Eigen::MatrixXd::Zero(dim, dim));

	for (int m = 0; m <= geo.M; ++m)
	{
		if (n_stokes == 1)
		{
			F_packed[m] = Fcos[m];
		}
		else if (n_stokes == 3)
		{
			for (int i = 0; i < geo.Ntheta; ++i)
			{
				for (int j = 0; j < geo.Ntheta; ++j)
				{
					int r = 3 * i; int c = 3 * j;

					F_packed[m].block<2, 2>(r, c) = Fcos[m].block<2, 2>(r, c);
					F_packed[m](r + 2, c + 2) = Fcos[m](r + 2, c + 2);
					
					F_packed[m].block<2, 1>(r, c + 2) = -Fsin[m].block<2, 1>(r, c + 2);
					F_packed[m].block<1, 2>(r + 2, c) = Fsin[m].block<1, 2>(r + 2, c);
				}
			}
		}
		else if (n_stokes == 4)
		{
			for (int i = 0; i < geo.Ntheta; ++i)
			{
				for (int j = 0; j < geo.Ntheta; ++j)
				{
					int r = 4 * i; int c = 4 * j;

					F_packed[m].block<2, 2>(r, c) = Fcos[m].block<2, 2>(r, c);
					F_packed[m].block<2, 2>(r + 2, c + 2) = Fcos[m].block<2, 2>(r + 2, c + 2);
					
					F_packed[m].block<2, 2>(r, c + 2) = -Fsin[m].block<2, 2>(r, c + 2);
					F_packed[m].block<2, 2>(r + 2, c) = Fsin[m].block<2, 2>(r + 2, c);
				}
			}
		}
	}

	return F_packed;
}

std::vector<Eigen::MatrixXd> reconstruct(const std::vector<Eigen::MatrixXd>& F_mode, const std::vector<double>& out_phi, int n_theta)
{
	int dim = F_mode[0].rows();
	int n_stokes = dim / n_theta;
	int n_out = out_phi.size();
	int M = F_mode.size() - 1;
	
	std::vector<Eigen::MatrixXd> f_phi(n_out, Eigen::MatrixXd::Zero(dim, dim));

	for(int l = 0; l < n_out; l++)
	{
		f_phi[l] += F_mode[0];

		for(int m = 1; m <= M; m++)
		{
			double cos_ml = std::cos(double(m) * out_phi[l]);
			double sin_ml = std::sin(double(m) * out_phi[l]);
			
			Eigen::MatrixXd m_eff = Eigen::MatrixXd::Zero(dim, dim);

			if (n_stokes == 1)
			{
				m_eff = F_mode[m] * cos_ml;
			}
			else if (n_stokes == 3)
			{
				for (int i = 0; i < n_theta; ++i)
				{
					for (int j = 0; j < n_theta; ++j)
					{
						int r = 3 * i; int c = 3 * j;

						m_eff.block<2, 2>(r, c) = F_mode[m].block<2, 2>(r, c) * cos_ml;
						m_eff(r+2, c+2) = F_mode[m](r+2, c+2) * cos_ml;
						
						m_eff.block<2, 1>(r, c+2) = -F_mode[m].block<2, 1>(r, c+2) * sin_ml;
						m_eff.block<1, 2>(r+2, c) = F_mode[m].block<1, 2>(r+2, c) * sin_ml;
					}
				}
			}
			else if (n_stokes == 4)
			{
				for (int i = 0; i < n_theta; ++i)
				{
					for (int j = 0; j < n_theta; ++j)
					{
						int r = 4 * i; int c = 4 * j;

						m_eff.block<2, 2>(r, c) = F_mode[m].block<2, 2>(r, c) * cos_ml;
						m_eff.block<2, 2>(r+2, c+2) = F_mode[m].block<2, 2>(r+2, c+2) * cos_ml;
						
						m_eff.block<2, 2>(r, c+2) = -F_mode[m].block<2, 2>(r, c+2) * sin_ml;
						m_eff.block<2, 2>(r+2, c) = F_mode[m].block<2, 2>(r+2, c) * sin_ml;
					}
				}
			}

			f_phi[l] += 2.0 * m_eff;
		}
	}

	return f_phi;
}

Geometry generateGeometryGaussRadau(int n_theta, int n_phi, int n_mode)
{
	Geometry geometry;
	
	geometry.Ntheta = n_theta;
	geometry.Nphi = (n_phi > 0) ? n_phi : n_theta * 4 + 1;
	geometry.M = (n_mode >= 0) ? n_mode : (geometry.Nphi - 3) / 2;
	
	geometry.theta_uh = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.theta_lh = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.theta_all = Eigen::VectorXd::Zero(geometry.Ntheta * 2);
	
	geometry.weight = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.mu = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.WMU = Eigen::MatrixXd::Zero(geometry.Ntheta, geometry.Ntheta);

	geometry.phi = Eigen::VectorXd::Zero(geometry.Nphi);
	geometry.d_phi = 2.0 * std::numbers::pi / double(geometry.Nphi);

	for(int i = 0; i < geometry.Nphi; i++)
	{
		geometry.phi(i) = double(i) * geometry.d_phi;
	}

	auto node_weight = math::computeGaussRadauQuadratureNodeWeight(geometry.Ntheta);

	for(int i = 0; i < node_weight.size(); i++)
	{
		double acos_val = std::acos(node_weight[node_weight.size() - 1 - i][0]);
		
		geometry.theta_uh(i) = acos_val;
		geometry.theta_lh(i) = std::numbers::pi - acos_val;

		geometry.theta_all(i) = geometry.theta_uh(i);
		geometry.theta_all(geometry.Ntheta * 2 - 1 - i) = geometry.theta_lh(i);

		geometry.mu(i) = node_weight[node_weight.size() - 1 - i][0];
		geometry.weight(i) = node_weight[node_weight.size() - 1 - i][1];
	}

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		geometry.WMU(i, i) = geometry.weight(i) * geometry.mu(i);
	}

	return geometry;
}

Geometry generateGeometryRegular(int n_theta, int n_phi, int n_mode, double eps)
{
	Geometry geometry;
	
	geometry.Ntheta = n_theta;
	geometry.Nphi = (n_phi > 0) ? n_phi : n_theta * 4 + 1;
	geometry.M = (n_mode >= 0) ? n_mode : (geometry.Nphi - 3) / 2;
	
	geometry.theta_uh = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.theta_lh = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.theta_all = Eigen::VectorXd::Zero(geometry.Ntheta * 2);
	
	geometry.weight = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.mu = Eigen::VectorXd::Zero(geometry.Ntheta);
	geometry.WMU = Eigen::MatrixXd::Zero(geometry.Ntheta, geometry.Ntheta);

	geometry.phi = Eigen::VectorXd::Zero(geometry.Nphi);
	geometry.d_phi = 2.0 * std::numbers::pi / double(geometry.Nphi);

	for(int i = 0; i < geometry.Nphi; i++)
	{
		geometry.phi(i) = double(i) * geometry.d_phi;
	}

	eps = std::numbers::pi / 2.0 / double(geometry.Ntheta - 1) * eps;
	double dtheta = (std::numbers::pi / 2.0 - eps) / double(geometry.Ntheta - 1);

	for(int i = 0; i < geometry.Ntheta; i++)
	{   
		geometry.theta_uh(i) = 0.0 + dtheta * double(i);
		geometry.theta_lh(i) = std::numbers::pi - dtheta * double(i);
		geometry.mu(i) = std::abs(std::cos(geometry.theta_uh(i)));

		geometry.theta_all(i) = geometry.theta_uh(i);
		geometry.theta_all(geometry.Ntheta * 2 - 1 - i) = geometry.theta_lh(i);
	}

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		if(i <= geometry.Ntheta - 3)
		{
			if(i == 0 || i == geometry.Ntheta - 3)
			{
				geometry.weight(i) = 1.0 / 3.0 * dtheta * std::sin(geometry.theta_uh(i));
			}
			else if(i % 2 == 1)
			{
				geometry.weight(i) = 4.0 / 3.0 * dtheta * std::sin(geometry.theta_uh(i));
			}
			else
			{
				geometry.weight(i) = 2.0 / 3.0 * dtheta * std::sin(geometry.theta_uh(i));
			}
		}

		if(i == geometry.Ntheta - 3)
		{
			geometry.weight(i) += ((1.0 / (6.0 * dtheta * dtheta)) * std::pow(2.0 * dtheta + eps, 3) - (3.0 / (4.0 * dtheta)) * std::pow(2.0 * dtheta + eps, 2) + (2.0 * dtheta + eps)) * std::sin(geometry.theta_uh(i));
		}
		else if(i == geometry.Ntheta - 2)
		{
			geometry.weight(i) = ((- 1.0 / (3.0 * dtheta * dtheta)) * std::pow(2.0 * dtheta + eps, 3) + (1.0 / dtheta) * std::pow(2.0 * dtheta + eps, 2)) * std::sin(geometry.theta_uh(i));
		}
		else if(i == geometry.Ntheta - 1)
		{
			geometry.weight(i) = ((1.0 / (6.0 * dtheta * dtheta)) * std::pow(2.0 * dtheta + eps, 3) - (1.0 / (4.0 * dtheta)) * std::pow(2.0 * dtheta + eps, 2)) * std::sin(geometry.theta_uh(i));
		}
	}

	for(int i = 0; i < geometry.Ntheta; i++)
	{
		geometry.WMU(i, i) = geometry.weight(i) * geometry.mu(i);
	}

	return geometry;
}

}
