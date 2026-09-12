#include "inversion.hpp"
#include "physics.hpp"
#include "mie.hpp"
#include "rayleigh.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace paad::core
{

AtmosphereSensitivity computeAtmosphericJacobian(const atmosphere::AtmosphereModel& atmos, const std::vector<OpticalSensitivity>& opt_sens_layers, const OpticalLayer& surface_opt_layer, const OpticalSensitivity& opt_sens_surface, double spectral, SpectralCoordinateDimension dim, int n_parallel)
{
	AtmosphereSensitivity atm_sens;
	int n_layer = atmos.layers.size();
	int n_species = atmos.species.size();

	atm_sens.temperature.assign(n_layer, 0.0);
	atm_sens.pressure.assign(n_layer, 0.0);
	atm_sens.number_density.assign(n_layer, 0.0);
	
	atm_sens.species.resize(n_species);

	for(int j = 0; j < n_species; ++j)
	{
		atm_sens.species[j].number_density.assign(n_layer, 0.0);
		atm_sens.species[j].mixing_ratio.assign(n_layer, 0.0);
		
		const auto& spec = atmos.species[j];
		
		if (std::holds_alternative<atmosphere::MieScattering>(spec.scattering_model))
		{
			MieSensitivity mie_sens;
			mie_sens.refractive_index_real.assign(n_layer, 0.0);
			mie_sens.refractive_index_imag.assign(n_layer, 0.0);
			auto& sd_model = std::get<atmosphere::MieScattering>(spec.scattering_model).size_distribution;
			
			if (std::holds_alternative<atmosphere::DeltaDistribution>(sd_model))
			{
				DeltaSensitivity d_sens; d_sens.r.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(d_sens);
			}
			else if (std::holds_alternative<atmosphere::LogNormalDistribution>(sd_model))
			{
				LogNormalSensitivity lnd; lnd.r_g.assign(n_layer, 0.0); lnd.sigma_g.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(lnd);
			}
			else if (std::holds_alternative<atmosphere::GammaDistribution>(sd_model))
			{
				GammaSensitivity gd; gd.a.assign(n_layer, 0.0); gd.b.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(gd);
			}
			else if (std::holds_alternative<atmosphere::RectangularDistribution>(sd_model))
			{
				RectangularSensitivity rd; rd.r_mean.assign(n_layer, 0.0); rd.width.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(rd);
			}
			else if (std::holds_alternative<atmosphere::ModifiedGammaDistribution>(sd_model))
			{
				ModifiedGammaSensitivity mgd; mgd.r_c.assign(n_layer, 0.0); mgd.alpha.assign(n_layer, 0.0); mgd.gamma.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(mgd);
			}
			else if (std::holds_alternative<atmosphere::PowerLawDistribution>(sd_model))
			{
				PowerLawSensitivity pld; pld.delta.assign(n_layer, 0.0); pld.r1.assign(n_layer, 0.0); pld.r2.assign(n_layer, 0.0);
				mie_sens.size_distribution = std::move(pld);
			}

			atm_sens.species[j].scattering = std::move(mie_sens);
		}
		else if (std::holds_alternative<atmosphere::RayleighScattering>(spec.scattering_model))
		{
			RayleighSensitivity ray; ray.refractive_index.assign(n_layer, 0.0); ray.depolarization_factor.assign(n_layer, 0.0); ray.number_density_reference.assign(n_layer, 0.0);
			atm_sens.species[j].scattering = std::move(ray);
		}
		else if (std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model))
		{
			HenyeyGreensteinSensitivity hg; hg.cross_section.assign(n_layer, 0.0); hg.asymmetry_factor.assign(n_layer, 0.0);
			atm_sens.species[j].scattering = std::move(hg);
		}
		else if (std::holds_alternative<atmosphere::ConstantScattering>(spec.scattering_model))
		{
			ConstantScatteringSensitivity cs; cs.cross_section.assign(n_layer, 0.0);
			atm_sens.species[j].scattering = std::move(cs);
		}
		else if (std::holds_alternative<atmosphere::ExternalScattering>(spec.scattering_model))
		{
			ExternalScatteringSensitivity ext; ext.cross_section.assign(n_layer, 0.0);
			atm_sens.species[j].scattering = std::move(ext);
		}
		else
		{
			atm_sens.species[j].scattering = std::monostate{};
		}

		if (std::holds_alternative<atmosphere::ConstantAbsorption>(spec.absorption_model))
		{
			ConstantAbsorptionSensitivity ca; ca.cross_section.assign(n_layer, 0.0);
			atm_sens.species[j].absorption = std::move(ca);
		}
		else
		{
			atm_sens.species[j].absorption = std::monostate{};
		}
	}

	double wavelength = (dim == SpectralCoordinateDimension::Wavelength) ? spectral : 1.0 / spectral;
	double wavenumber = (dim == SpectralCoordinateDimension::Wavenumber) ? spectral : 1.0 / spectral;

	atm_sens.surface.albedo = opt_sens_surface.surface_albedo;
	atm_sens.surface.emissivity = opt_sens_surface.surface_emissivity;
	double dB_dT = physics::computePlanckFunctionDerivative(spectral, atmos.surface.temperature, dim);
	double dJ_dB_surf = opt_sens_surface.planck_function; 
	atm_sens.surface.temperature = dJ_dB_surf * dB_dT;

	#pragma omp parallel for num_threads(n_parallel)
	for (int i = 0; i < n_layer; ++i)
	{
		const auto& layer = atmos.layers[i];
		const auto& opt_sens = opt_sens_layers[i];
		double dz = layer.altitude_top - layer.altitude_bottom;
		int n_angle = opt_sens.scattering_phase_matrix.size();

		if (opt_sens.planck_function != 0.0)
		{
			double dB_dT_layer = physics::computePlanckFunctionDerivative(spectral, layer.temperature, dim);
			atm_sens.temperature[i] += opt_sens.planck_function * dB_dT_layer;
		}

		double k_sca_tot = 0.0;
		double k_abs_tot = 0.0;
		std::vector<double> sig_sca_j(n_species, 0.0);
		std::vector<double> sig_abs_j(n_species, 0.0);
		std::vector<std::vector<Eigen::Matrix4d>> P_j(n_species, std::vector<Eigen::Matrix4d>(n_angle, Eigen::Matrix4d::Zero()));

		hitran::Diluent cached_diluent(1.0, 0.0, 0.0, 0.0, 0.0);

		if (!atmos.background_gas.diluent_species.empty())
		{
			double air=0, co2=0, h2=0, he=0, h2o=0;
			const auto& hc = atmos.background_gas;

			for (size_t d = 0; d < hc.diluent_species.size(); ++d)
			{
				std::string sp = hc.diluent_species[d];
				std::transform(sp.begin(), sp.end(), sp.begin(), ::toupper);

				if (sp == "AIR")
				{
					air = hc.diluent_ratio[d];
				}
				else if (sp == "CO2")
				{
					co2 = hc.diluent_ratio[d];
				}
				else if (sp == "H2")
				{
					h2 = hc.diluent_ratio[d];
				}
				else if (sp == "HE")
				{
					he = hc.diluent_ratio[d];
				}
				else if (sp == "H2O")
				{
					h2o = hc.diluent_ratio[d];
				}
			}

			cached_diluent = hitran::Diluent(air, co2, h2, he, h2o);
		}

		for (int j = 0; j < n_species; ++j)
		{
			const auto& spec = atmos.species[j];
			
			if (std::holds_alternative<atmosphere::MieScattering>(spec.scattering_model))
			{
				auto& mie_scat = std::get<atmosphere::MieScattering>(spec.scattering_model);
				autodiff::complex<double> m(mie_scat.refractive_index[0], mie_scat.refractive_index[1]);
				auto res = mie::computeMieScatteringSizeDistribution(n_angle, wavelength, mie_scat.weight_particle_size_distribution, m);
				sig_sca_j[j] = res.scattering_cross_section();
				sig_abs_j[j] += res.absorption_cross_section(); 

				for(int k = 0; k < n_angle; ++k)
				{
					P_j[j][k] = res.scattering_matrix(k);
				}
			}
			else if (std::holds_alternative<atmosphere::RayleighScattering>(spec.scattering_model))
			{
				auto& ray = std::get<atmosphere::RayleighScattering>(spec.scattering_model);
				auto res = rayleigh::computeRayleighScattering(n_angle, wavelength, ray.refractive_index[0], ray.number_density_reference, ray.depolarization_factor);
				sig_sca_j[j] = res.scattering_cross_section();
				sig_abs_j[j] += res.absorption_cross_section(); 

				for(int k = 0; k < n_angle; ++k)
				{
					P_j[j][k] = res.scattering_matrix(k);
				}
			}
			else if (std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model))
			{
				auto& hg = std::get<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model);
				sig_sca_j[j] = hg.cross_section;
				double g = hg.asymmetry_factor;

				for(int k = 0; k < n_angle; ++k)
				{
					double theta = opt_sens.scattering_phase_matrix[k].first;
					P_j[j][k](0, 0) = (1.0 - g * g) / std::pow(1.0 + g * g - 2.0 * g * std::cos(theta), 1.5);
				}
			}
			else if (std::holds_alternative<atmosphere::ConstantScattering>(spec.scattering_model))
			{
				sig_sca_j[j] = std::get<atmosphere::ConstantScattering>(spec.scattering_model).cross_section;

				for(int k = 0; k < n_angle; ++k)
				{
					P_j[j][k](0, 0) = 1.0;
				}
			}
			else if (std::holds_alternative<atmosphere::ExternalScattering>(spec.scattering_model))
			{
				auto& ext = std::get<atmosphere::ExternalScattering>(spec.scattering_model);
				sig_sca_j[j] = ext.cross_section;

				for(int k = 0; k < n_angle; ++k)
				{
					double theta = opt_sens.scattering_phase_matrix[k].first;

					for (int r = 0; r < 4; ++r)
					{
						for (int c = 0; c < 4; ++c)
						{
							P_j[j][k](r, c) = ext.model.interpolateElement(theta, r, c);
						}
					}
				}
			}

			if (std::holds_alternative<atmosphere::ConstantAbsorption>(spec.absorption_model))
			{
				sig_abs_j[j] += std::get<atmosphere::ConstantAbsorption>(spec.absorption_model).cross_section;
			}
			else if (std::holds_alternative<atmosphere::ArrheniusAbsorption>(spec.absorption_model))
			{
				sig_abs_j[j] += std::get<atmosphere::ArrheniusAbsorption>(spec.absorption_model).model.cross_section(spectral, layer.temperature);
			}
			else if (std::holds_alternative<atmosphere::HitranAbsorption>(spec.absorption_model))
			{
				auto& hit_abs = std::get<atmosphere::HitranAbsorption>(spec.absorption_model);
				double pressure_self = layer.pressure * spec.vertical_mixing_ratio_profile[i];
				double sum_abs = 0.0;

				for (const auto& line : hit_abs.cached_lines)
				{
					sum_abs += line.computeCrossSection(wavenumber, layer.temperature, pressure_self, layer.pressure, hit_abs.isotopologue_data, cached_diluent);
				}

				sig_abs_j[j] += sum_abs;
			}

			double N_j = spec.vertical_number_density_profile[i];
			k_sca_tot += sig_sca_j[j] * N_j;
			k_abs_tot += sig_abs_j[j] * N_j;
		}

		double k_ext_tot = k_sca_tot + k_abs_tot;
		double omega = (k_ext_tot > 1e-30) ? k_sca_tot / k_ext_tot : 0.0;

		std::vector<Eigen::Matrix4d> P_tot(n_angle, Eigen::Matrix4d::Zero());

		if (k_sca_tot > 1e-30)
		{
			for(int k = 0; k < n_angle; ++k)
			{
				for(int j = 0; j < n_species; ++j)
				{
					P_tot[k] += (sig_sca_j[j] * atmos.species[j].vertical_number_density_profile[i]) * P_j[j][k];
				}

				P_tot[k] /= k_sca_tot;
			}
		}

		auto apply_chain_rule = [&](double d_ksca, double d_kabs, const std::vector<Eigen::Matrix4d>& d_P, int spec_idx) -> double
		{
			double d_kext = d_ksca + d_kabs;
			double d_tau = d_kext * dz;
			double d_omega = 0.0;

			if (k_ext_tot > 1e-30)
			{
				d_omega = (d_ksca - omega * d_kext) / k_ext_tot;
			}
			
			double grad = 0.0;
			grad += opt_sens.optical_thickness * d_tau;
			grad += opt_sens.single_scattering_albedo * d_omega;
			
			if (k_sca_tot > 1e-30)
			{
				double N_j = atmos.species[spec_idx].vertical_number_density_profile[i];
				double k_sca_j_val = sig_sca_j[spec_idx] * N_j;

				for(int k = 0; k < n_angle; ++k)
				{
					Eigen::Matrix4d dP_tot = (k_sca_j_val * d_P[k] + d_ksca * (P_j[spec_idx][k] - P_tot[k])) / k_sca_tot;
					grad += (opt_sens.scattering_phase_matrix[k].second.array() * dP_tot.array()).sum();
				}
			}

			return grad;
		};

		std::vector<Eigen::Matrix4d> d_P_zero(n_angle, Eigen::Matrix4d::Zero());

		for (int j = 0; j < n_species; ++j)
		{
			const auto& spec = atmos.species[j];
			double N_j = spec.vertical_number_density_profile[i];
			double N_tot = layer.number_density;

			double grad_N_j = apply_chain_rule(sig_sca_j[j], sig_abs_j[j], d_P_zero, j);
			atm_sens.species[j].number_density[i] = grad_N_j;

			if (N_tot > 1e-30)
			{
				atm_sens.species[j].mixing_ratio[i] = grad_N_j * N_tot;
			}

			if (std::holds_alternative<atmosphere::MieScattering>(spec.scattering_model))
			{
				auto& mie_scat = std::get<atmosphere::MieScattering>(spec.scattering_model);
				autodiff::complex<double> m(mie_scat.refractive_index[0], mie_scat.refractive_index[1]);
				
				mie::MieResult<double> res_y;
				mie::DiffFlags flags;
				flags.n_r = true;
				flags.n_i = true;

				auto extract_mie_grad = [&](const mie::MieJacobian& jac, int col, std::vector<double>& target_arr)
				{
					double d_ksca = N_j * jac.scattering_cross_section(col);
					double d_kabs = N_j * jac.absorption_cross_section(col);
					std::vector<Eigen::Matrix4d> d_P(n_angle);

					for(int k = 0; k < n_angle; ++k)
					{
						d_P[k] = jac.scattering_matrix(k, col);
					}

					target_arr[i] = apply_chain_rule(d_ksca, d_kabs, d_P, j);
				};

				int n_sampling = mie_scat.particle_size_distribution.size();
				auto& mie_sens = std::get<MieSensitivity>(atm_sens.species[j].scattering);

				if (std::holds_alternative<atmosphere::DeltaDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::DeltaDistribution>(mie_scat.size_distribution);
					flags.delta_r = true;
					auto jac = mie::computeDeltaMieJacobian(n_angle, wavelength, d.r, m, res_y, flags);
					auto& d_sens = std::get<DeltaSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.r);
					extract_mie_grad(jac, 1, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 2, mie_sens.refractive_index_imag);
				}
				else if (std::holds_alternative<atmosphere::LogNormalDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::LogNormalDistribution>(mie_scat.size_distribution);
					flags.lnd_r_g = true; flags.lnd_sigma_g = true;
					auto jac = mie::computeLogNormalMieJacobian(n_angle, n_sampling, wavelength, d.r_g, d.sigma_g, d.r_min, d.r_max, m, res_y, flags);
					auto& d_sens = std::get<LogNormalSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.r_g);
					extract_mie_grad(jac, 1, d_sens.sigma_g);
					extract_mie_grad(jac, 2, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 3, mie_sens.refractive_index_imag);
				}
				else if (std::holds_alternative<atmosphere::GammaDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::GammaDistribution>(mie_scat.size_distribution);
					flags.gd_a = true; flags.gd_b = true;
					auto jac = mie::computeGammaMieJacobian(n_angle, n_sampling, wavelength, d.a, d.b, m, res_y, flags);
					auto& d_sens = std::get<GammaSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.a);
					extract_mie_grad(jac, 1, d_sens.b);
					extract_mie_grad(jac, 2, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 3, mie_sens.refractive_index_imag);
				}
				else if (std::holds_alternative<atmosphere::RectangularDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::RectangularDistribution>(mie_scat.size_distribution);
					flags.rect_r_mean = true; flags.rect_width = true;
					auto jac = mie::computeRectangularMieJacobian(n_angle, n_sampling, wavelength, d.r_mean, d.width, m, res_y, flags);
					auto& d_sens = std::get<RectangularSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.r_mean);
					extract_mie_grad(jac, 1, d_sens.width);
					extract_mie_grad(jac, 2, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 3, mie_sens.refractive_index_imag);
				}
				else if (std::holds_alternative<atmosphere::ModifiedGammaDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::ModifiedGammaDistribution>(mie_scat.size_distribution);
					flags.mgd_r_c = true; flags.mgd_alpha = true; flags.mgd_gamma = true;
					auto jac = mie::computeModifiedGammaMieJacobian(n_angle, n_sampling, wavelength, d.r_c, d.alpha, d.gamma, m, res_y, flags);
					auto& d_sens = std::get<ModifiedGammaSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.r_c);
					extract_mie_grad(jac, 1, d_sens.alpha);
					extract_mie_grad(jac, 2, d_sens.gamma);
					extract_mie_grad(jac, 3, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 4, mie_sens.refractive_index_imag);
				}
				else if (std::holds_alternative<atmosphere::PowerLawDistribution>(mie_scat.size_distribution))
				{
					auto& d = std::get<atmosphere::PowerLawDistribution>(mie_scat.size_distribution);
					flags.pld_delta = true; flags.pld_r1 = true; flags.pld_r2 = true;
					auto jac = mie::computePowerLawMieJacobian(n_angle, n_sampling, wavelength, d.delta, d.r1, d.r2, m, res_y, flags);
					auto& d_sens = std::get<PowerLawSensitivity>(mie_sens.size_distribution);

					extract_mie_grad(jac, 0, d_sens.delta);
					extract_mie_grad(jac, 1, d_sens.r1);
					extract_mie_grad(jac, 2, d_sens.r2);
					extract_mie_grad(jac, 3, mie_sens.refractive_index_real);
					extract_mie_grad(jac, 4, mie_sens.refractive_index_imag);
				}
			}
			else if (std::holds_alternative<atmosphere::RayleighScattering>(spec.scattering_model))
			{
				auto& ray = std::get<atmosphere::RayleighScattering>(spec.scattering_model);
				rayleigh::RayleighResult<double> res_y;
				rayleigh::DiffFlags flags;
				flags.n_r = true;
				flags.depolarization_factor = true;
				auto jac = rayleigh::computeRayleighJacobian(n_angle, wavelength, ray.refractive_index[0], ray.number_density_reference, ray.depolarization_factor, res_y, flags);

				auto extract_ray_grad = [&](const rayleigh::RayleighJacobian& jac_in, int col, std::vector<double>& target_arr)
				{
					double d_ksca = N_j * jac_in.scattering_cross_section(col);
					double d_kabs = N_j * jac_in.absorption_cross_section(col);
					std::vector<Eigen::Matrix4d> d_P(n_angle);

					for(int k = 0; k < n_angle; ++k)
					{
						d_P[k] = jac_in.scattering_matrix(k, col);
					}

					target_arr[i] = apply_chain_rule(d_ksca, d_kabs, d_P, j);
				};

				auto& ray_sens = std::get<RayleighSensitivity>(atm_sens.species[j].scattering);
				extract_ray_grad(jac, 0, ray_sens.refractive_index);
				extract_ray_grad(jac, 1, ray_sens.depolarization_factor);

				double d_sigma_dNref = -2.0 * sig_sca_j[j] / ray.number_density_reference;
				ray_sens.number_density_reference[i] = apply_chain_rule(d_sigma_dNref * N_j, 0.0, d_P_zero, j);
			}
			else if (std::holds_alternative<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model))
			{
				auto& hg = std::get<atmosphere::HenyeyGreensteinScattering>(spec.scattering_model);
				auto& hg_sens = std::get<HenyeyGreensteinSensitivity>(atm_sens.species[j].scattering);
				hg_sens.cross_section[i] = apply_chain_rule(N_j, 0.0, d_P_zero, j);

				double g = hg.asymmetry_factor;
				std::vector<Eigen::Matrix4d> d_P_dg(n_angle, Eigen::Matrix4d::Zero())

				for(int k = 0; k < n_angle; ++k)
				{
					double theta = opt_sens.scattering_phase_matrix[k].first;
					double cos_t = std::cos(theta);
					double denom = 1.0 + g*g - 2.0*g*cos_t;
					d_P_dg[k](0,0) = (-2.0 * g / std::pow(denom, 1.5)) - 1.5 * (1.0 - g*g) * (2.0*g - 2.0*cos_t) / std::pow(denom, 2.5);
				}

				hg_sens.asymmetry_factor[i] = apply_chain_rule(0.0, 0.0, d_P_dg, j);
			}
			else if (std::holds_alternative<atmosphere::ConstantScattering>(spec.scattering_model))
			{
				auto& iso_sens = std::get<ConstantScatteringSensitivity>(atm_sens.species[j].scattering);
				iso_sens.cross_section[i] = apply_chain_rule(N_j, 0.0, d_P_zero, j);
			}
			else if (std::holds_alternative<atmosphere::ExternalScattering>(spec.scattering_model))
			{
				auto& ext_sens = std::get<ExternalScatteringSensitivity>(atm_sens.species[j].scattering);
				ext_sens.cross_section[i] = apply_chain_rule(N_j, 0.0, d_P_zero, j);
			}

			if (std::holds_alternative<atmosphere::ConstantAbsorption>(spec.absorption_model))
			{
				auto& abs_sens = std::get<ConstantAbsorptionSensitivity>(atm_sens.species[j].absorption);
				abs_sens.cross_section[i] = apply_chain_rule(0.0, N_j, d_P_zero, j);
			}
			else if (std::holds_alternative<atmosphere::ArrheniusAbsorption>(spec.absorption_model))
			{
				auto& arr = std::get<atmosphere::ArrheniusAbsorption>(spec.absorption_model);
				autodiff::dual<double> T_ad(layer.temperature, 1.0);
				autodiff::dual<double> abs_ad = arr.model.cross_section(spectral, T_ad);
				
				double d_sigma_dT = abs_ad.der;
				atm_sens.temperature[i] += apply_chain_rule(0.0, N_j * d_sigma_dT, d_P_zero, j);
			}
			else if (std::holds_alternative<atmosphere::HitranAbsorption>(spec.absorption_model))
			{
				auto& hit = std::get<atmosphere::HitranAbsorption>(spec.absorption_model);
				autodiff::dual<double> T_ad(layer.temperature, 0.0);
				autodiff::dual<double> P_ad(layer.pressure, 0.0);
				double c_j = spec.vertical_mixing_ratio_profile[i];
				auto Pself_ad = P_ad * c_j;

				auto run_pass = [&](double t_seed, double p_seed, double& d_dT, double& d_dP)
				{
					T_ad.der = t_seed;
					P_ad.der = p_seed;
					Pself_ad = P_ad * c_j;
					autodiff::dual<double> sum_abs_ad = 0.0;

					for (const auto& line : hit.cached_lines)
					{
						sum_abs_ad += line.computeCrossSection(wavenumber, T_ad, Pself_ad, P_ad, hit.isotopologue_data, cached_diluent);
					}

					if (t_seed > 0)
					{
						d_dT = sum_abs_ad.der;
					}

					if (p_seed > 0)
					{
						d_dP = sum_abs_ad.der;
					}
				};

				double d_sigma_dT = 0.0, d_sigma_dP = 0.0;
				run_pass(1.0, 0.0, d_sigma_dT, d_sigma_dP);
				run_pass(0.0, 1.0, d_sigma_dT, d_sigma_dP);

				atm_sens.temperature[i] += apply_chain_rule(0.0, N_j * d_sigma_dT, d_P_zero, j); 
				atm_sens.pressure[i] += apply_chain_rule(0.0, N_j * d_sigma_dP, d_P_zero, j); 
			}
		}
	}

	return atm_sens;
}

}
