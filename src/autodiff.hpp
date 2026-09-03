#pragma once

#include <iostream>
#include <cmath>
#include <complex>

namespace autodiff
{
// 双対数構造体
template <typename T> struct dual
{
	T val; // 実数部
	T der; // 双対部

	// コンストラクタ
	dual(T v = 0.0, T d = 0.0) : val(v), der(d)
	{
		;
	}

	explicit operator T() const;

	dual operator+(const dual&) const;
	dual operator-(const dual&) const;
	dual operator-() const;
	dual operator*(const dual&) const;
	dual operator/(const dual&) const;

	dual& operator+=(const dual<T>&);
	dual& operator-=(const dual<T>&);
	dual& operator*=(const dual<T>&);
	dual& operator/=(const dual<T>&);

	dual& operator+=(const T&);
	dual& operator-=(const T&);
	dual& operator*=(const T&);
	dual& operator/=(const T&);
};

template <typename T> inline dual<T>::operator T() const
{
	return val;
}

template <typename T> inline dual<T> dual<T>::operator+(const dual<T>& rhs) const
{
	return dual<T>(val + rhs.val, der + rhs.der);
}

template <typename T> inline dual<T> dual<T>::operator-(const dual<T>& rhs) const
{
	return dual<T>(val - rhs.val, der - rhs.der);
}

template <typename T> inline dual<T> dual<T>::operator-() const
{
	return dual<T>(-val, -der);
}

template <typename T> inline dual<T> dual<T>::operator*(const dual<T>& rhs) const
{
	return dual<T>(val * rhs.val, val * rhs.der + der * rhs.val);
}

template <typename T> inline dual<T> dual<T>::operator/(const dual<T>& rhs) const
{
	return dual<T>(val / rhs.val, (der * rhs.val - val * rhs.der) / (rhs.val * rhs.val));
}

template <typename T> inline auto get_value(const T& x)
{
	if constexpr(requires {x.val; x.der;})
	{
		return x.val;
	} 
	else 
	{
		return x;
	}
}

template <typename T> inline dual<T> sin(const dual<T>& x)
{
	using std::sin; using std::cos;
	return dual<T>(sin(x.val), cos(x.val) * x.der);
}

template <typename T> inline dual<T> cos(const dual<T>& x)
{
	using std::sin; using std::cos;
	return dual<T>(cos(x.val), -sin(x.val) * x.der);
}

template <typename T> inline dual<T> tan(const dual<T>& x)
{
	using std::tan; using std::cos;
	return dual<T>(tan(x.val), x.der / cos(x.val) / cos(x.val));
}

template <typename T> inline dual<T> asin(const dual<T>& x)
{
	// d/dx(asin(x)) = 1 / sqrt(1 - x^2)
	using std::asin; using std::sqrt;
	return dual<T>(asin(x.val), x.der / sqrt(T(1.0) - x.val * x.val));
}

template <typename T> inline dual<T> acos(const dual<T>& x)
{
	// d/dx(acos(x)) = -1 / sqrt(1 - x^2)
	using std::asin; using std::sqrt;
	return dual<T>(acos(x.val), -x.der / sqrt(T(1.0) - x.val * x.val));
}

template <typename T> inline dual<T> atan(const dual<T>& x)
{
	// d/dx(atan(x)) = 1 / (1 + x^2)
	using std::atan;
	return dual<T>(atan(x.val), x.der / (T(1.0) + x.val * x.val));
}

template <typename T> inline dual<T> atan2(const dual<T>& y, const dual<T>& x)
{
	using std::atan2;
	T val = atan2(y.val, x.val);
	T denom = x.val * x.val + y.val * y.val;
	T der = (y.der * x.val - x.der * y.val) / denom;
	
	return dual<T>(val, der);
}

template <typename T> inline dual<T> sinh(const dual<T>& x)
{
	using std::sinh; using std::cosh;
	return dual<T>(sinh(x.val), cosh(x.val) * x.der);
}

template <typename T> inline dual<T> cosh(const dual<T>& x)
{
	using std::sinh; using std::cosh;
	return dual<T>(cosh(x.val), sinh(x.val) * x.der);
}

template <typename T> inline dual<T> tanh(const dual<T>& x)
{
	using std::tanh;
	T th = tanh(x.val);
	return dual<T>(th, x.der * (T(1.0) - th * th));
}

template <typename T> inline dual<T> exp(const dual<T>& x)
{
	using std::exp;
	T e = exp(x.val);
	return dual<T>(e, e * x.der);
}

template <typename T> inline dual<T> log(const dual<T>& x)
{
	using std::log;
	return dual<T>(log(x.val), x.der / x.val);
}

template <typename T> inline dual<T> cbrt(const dual<T>& x)
{
	using std::cbrt;
	T res_val = cbrt(x.val);
	return dual<T>(res_val, x.der / (T(3.0) * res_val * res_val));
}

template <typename T> inline dual<T> abs(const dual<T>& x)
{
	using std::abs;
	T sign = (x.val > 0.0) ? T(1.0) : ((x.val < 0.0) ? T(-1.0) : T(0.0));
	return dual<T>(abs(x.val), x.der * sign);
}

template <typename T> inline dual<T> sqrt(const dual<T>& x)
{
	using std::sqrt;
	return dual<T>(sqrt(x.val), x.der / (T(2.0) * sqrt(x.val)));
}

template <typename T> inline dual<T> pow(const dual<T>& base, const dual<T>& exp)
{
	using std::pow; using std::log;
	// dual^dual : d/dx(u^v) = v*u^(v-1)*u' + u^v*ln(u)*v'
	T val = pow(base.val, exp.val);
	T der = exp.val * pow(base.val, exp.val - T(1.0)) * base.der + val * log(base.val) * exp.der;
	return dual<T>(val, der);
}

template <typename T> inline dual<T> pow(const dual<T>& base, const T& exp)
{
	// dual^scalar
	return pow(base, dual<T>(exp, T(0.0)));
}

template <typename T> inline dual<T> pow(const T& base, const dual<T>& exp)
{
	// scalar^dual
	return pow(dual<T>(base, T(0.0)), exp);
}

template <typename T> inline dual<T> expm1(const dual<T>& x)
{
	using std::exp; using std::expm1;
	return dual<T>(expm1(x.val), exp(x.val) * x.der);
}

template <typename T> inline dual<T> log1p(const dual<T>& x)
{
	using std::log1p;
	return dual<T>(log1p(x.val), x.der / (T(1.0) + x.val));
}

template <typename T> inline dual<T> log10(const dual<T>& x)
{
	using std::log10; using std::log;
	return dual<T>(log10(x.val), x.der / (x.val * log(T(10.0))));
}

template <typename T> inline dual<T> log2(const dual<T>& x)
{
	using std::log2; using std::log;
	return dual<T>(log2(x.val), x.der / (x.val * log(T(2.0))));
}

template <typename T> inline dual<T> hypot(const dual<T>& x, const dual<T>& y)
{
	using std::hypot;
	T res_val = hypot(x.val, y.val);
	return dual<T>(res_val, (x.val * x.der + y.val * y.der) / res_val);
}

static const double digamma_P[] = {
	0.25479851061131551,
	-0.32555031186804491,
	-0.65031853770896507,
	-0.28919126444774784,
	-0.045251321448739056,
	-0.0020713321167745952
};

static const double digamma_Q[] = {
	1.0,
	2.0767117023730469,
	1.4606242909763515,
	0.43593529692665969,
	0.054151797245674225,
	0.0021284987017821144,
	-0.55789841321675513e-6
};

static const double digamma_asym_P[] = {
	0.083333333333333333,
	-0.0083333333333333333,
	0.0039682539682539683,
	-0.0041666666666666667,
	0.0075757575757575758,
	-0.021092796092796093,
	0.083333333333333333,
	-0.443259803921568627
};

template <typename T> inline T evaluate_poly(const double* coeffs, T z, int count)
{
	T sum = T(coeffs[count - 1]);

	for (int i = count - 2; i >= 0; --i)
	{
		sum = sum * z + T(coeffs[i]);
	}

	return sum;
}

template <typename T> inline T digamma(T x)
{
	using std::log; using std::tan; using std::floor;
	const double pi = std::numbers::pi;
	double vx = get_value(x);

	T result = T(0.0);

	if (vx <= -1.0)
	{
		T target = T(1.0) - x;
		T remainder = target - floor(get_value(target));

		if (get_value(remainder) > 0.5)
		{
			remainder -= T(1.0);
		}
		
		if (get_value(remainder) == 0.0)
		{
			return T(std::numeric_limits<double>::quiet_NaN());
		}
		
		return T(pi) / tan(T(pi) * remainder) + digamma(target);
	}
	
	if (vx == 0.0)
	{
		return T(std::numeric_limits<double>::quiet_NaN());
	}

	if (vx >= 10.0)
	{
		asymptotic:
		
		T x_minus_1 = x - T(1.0);
		T inv_x = T(1.0) / x_minus_1;
		T z = inv_x * inv_x;
		return result + log(x_minus_1) + T(0.5) * inv_x - z * evaluate_poly(digamma_asym_P, z, 8);
	}

	while (get_value(x) > 2.0)
	{
		if (get_value(x) >= 10.0)
		{
			goto asymptotic;
		}

		x -= T(1.0);
		result += T(1.0) / x;
	}

	while (get_value(x) < 1.0)
	{
		result -= T(1.0) / x;
		x += T(1.0);
	}

	static const float Y = 0.99558162689208984F;
	static const double root1 = 1569415565.0 / 1073741824.0;
	static const double root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
	static const double root3 = 0.90163120932586959186e-19;

	T g = (x - T(root1)) - T(root2) - T(root3);
	T r = evaluate_poly(digamma_P, T(x - T(1.0)), 6) / evaluate_poly(digamma_Q, T(x - T(1.0)), 7);
	
	return result + g * T(Y) + g * r;
}

template <typename T> inline dual<T> tgamma(const dual<T>& x)
{
	using std::tgamma;
	T val = tgamma(x.val);
	return dual<T>(val, val * digamma(T(x.val)) * x.der);
}

template <typename T> inline dual<T> lgamma(const dual<T>& x)
{
	using std::lgamma;
	return dual<T>(lgamma(x.val), digamma(T(x.val)) * x.der);
}

template <typename T> inline dual<T> riemann_zeta(const dual<T>& s)
{
	using std::riemann_zeta; using std::log; using std::pow; using std::abs;
	T vs = s.val;
	T val = T(riemann_zeta(get_value(vs)));
	
	T eta_der = T(0.0);
	for(int n = 2; n < 1000; ++n)
	{
		T term = (n % 2 == 0 ? 1.0 : -1.0) * log(T(double(n))) / pow(T(double(n)), vs);
		eta_der += term;

		if (abs(get_value(term)) < 1e-18)
		{
			break;
		}
	}

	T f = T(1.0) - pow(T(2.0), T(1.0) - vs);
	T f_der = pow(T(2.0), T(1.0) - vs) * log(T(2.0));
	
	T der;

	if (abs(get_value(vs) - 1.0) < 1e-4)
	{
		der = T(-1.0) / ((vs - T(1.0)) * (vs - T(1.0)));
	}
	else
	{
		der = (eta_der - f_der * val) / f;
	}

	return dual<T>(val, der * s.der);
}

template <typename T> inline dual<T> beta(const dual<T>& a, const dual<T>& b)
{
	using std::beta;

	T val = beta(a.val, b.val);
	T psi_ab = digamma(a.val + b.val);

	return dual<T>(val, val * (digamma(a.val) - psi_ab) * a.der + val * (digamma(b.val) - psi_ab) * b.der);
}

template <typename T> inline dual<T> legendre(unsigned n, const dual<T>& x)
{
	using std::legendre;

	T val = legendre(n, x.val);
	
	if (n == 0)
	{
		return dual<T>(val, T(0.0));
	}

	T der = T(double(n)) / (x.val * x.val - T(1.0)) * (x.val * val - legendre(n - 1, x.val));

	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> assoc_legendre(unsigned n, unsigned m, const dual<T>& x)
{
	using std::assoc_legendre;

	T val = assoc_legendre(n, m, x.val);
	T der = (T(double(n)) * x.val * val - T(double(n + m)) * assoc_legendre(n - 1, m, x.val)) / (x.val * x.val - T(1.0));
	
	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> laguerre(unsigned n, const dual<T>& x)
{
	using std::laguerre;

	T val = laguerre(n, x.val);

	if (n == 0)
	{
		return dual<T>(val, T(0.0));
	}

	T der = (T(double(n)) * val - T(double(n)) * laguerre(n - 1, x.val)) / x.val;

	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> assoc_laguerre(unsigned n, unsigned m, const dual<T>& x)
{
	using std::assoc_laguerre;

	T val = assoc_laguerre(n, m, x.val);
	T der = (n == 0) ? T(0.0) : -assoc_laguerre(n - 1, m + 1, x.val);

	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> hermite(unsigned n, const dual<T>& x)
{
	using std::hermite;

	T val = hermite(n, x.val);
	T der = (n == 0) ? T(0.0) : T(2.0 * n) * hermite(n - 1, x.val);

	return dual<T>(val, der * x.der);
}

template <typename T> struct bessel_der_pair
{
	T d1;
	T d2;
};

template <typename T> inline bessel_der_pair<T> bessel_nu_der_full(T nu, T x, int type)
{
	using std::log; using std::pow; using std::tgamma; using std::abs;

	T x2 = x * 0.5;
	T l_x2 = log(x2);
	T sum1 = 0, sum2 = 0;
	
	for (int k = 0; k < 65; ++k)
	{
		T arg = nu + T(double(k + 1));
		double v_arg = get_value(arg);
		T term, g1, g2;

		if (v_arg <= 0 && std::abs(v_arg - std::round(v_arg)) < 1e-12)
		{
			int m = (int)std::abs(std::round(v_arg));

			T factor = pow(x2, nu + T(double(2 * k))) * (T(double((k % 2 == 0 || type == 1) ? 1 : -1)) / T(double(tgamma(k + 1))));
			T d1_gamma = T(double((m % 2 == 0) ? 1 : -1) * std::tgamma(m + 1));
			T d2_gamma = -2.0 * d1_gamma * digamma(T(double(m + 1)));
			g1 = d1_gamma; 
			g2 = 2.0 * d1_gamma * l_x2 + d2_gamma; 
			term = factor; 
		}
		else
		{
			T gamma_val = T(tgamma(v_arg));
			T common = pow(x2, nu + T(double(2 * k))) * (T(double((k % 2 == 0 || type == 1) ? 1 : -1)) / (T(double(tgamma(k + 1))) * gamma_val));
			T psi0 = digamma(arg);
			T psi1 = 0;

			for(int m=0; m<40; ++m)
			{
				psi1 += 1.0 / ((arg + T(double(m))) * (arg + T(double(m))));
			}

			g1 = l_x2 - psi0;
			g2 = g1 * g1 - psi1;
			term = common;
		}

		sum1 += term * g1;
		sum2 += term * g2;

		if (k > 10 && abs(get_value(term * g1)) < 1e-20)
		{
			break;
		}
	}

	return {sum1, sum2};
}

template <typename T> inline dual<T> cyl_bessel_j(const dual<T>& nu, const dual<T>& x)
{
	using std::cyl_bessel_j; using std::cyl_neumann;

	T v_nu = nu.val;
	T v_x = x.val;

	double dv = get_value(v_nu); double dx_v = get_value(v_x);

	T val = T(cyl_bessel_j(dv, dx_v));
	T dx = (v_nu / v_x) * val - T(cyl_bessel_j(dv + 1.0, dx_v));

	T dnu;
	if (dx_v > 30.0)
	{
		dnu = (std::numbers::pi * 0.5) * T(cyl_neumann(dv, dx_v));
	}
	else 
	{
		dnu = bessel_nu_der_full(v_nu, v_x, -1).d1;
	}

	return dual<T>(val, dnu * nu.der + dx * x.der);
}

template <typename T> inline dual<T> cyl_bessel_i(const dual<T>& nu, const dual<T>& x)
{
	using std::cyl_bessel_i;

	T v_nu = nu.val;
	T v_x = x.val;

	double dv = get_value(v_nu); double dx_v = get_value(v_x);

	T val = T(cyl_bessel_i(dv, dx_v));
	T dx = (v_nu / v_x) * val + T(cyl_bessel_i(dv + 1.0, dx_v));

	T dnu = bessel_nu_der_full(v_nu, v_x, 1).d1;

	return dual<T>(val, dnu * nu.der + dx * x.der);
}

template <typename T> inline dual<T> cyl_neumann(const dual<T>& nu, const dual<T>& x)
{
	using std::cyl_neumann; using std::cyl_bessel_j; using std::sin; using std::cos; using std::tan;

	T v_nu = nu.val; T v_x = x.val;
	double dv = get_value(v_nu); double dx_v = get_value(v_x);
	T val = T(cyl_neumann(dv, dx_v));
	T dx = (v_nu / v_x) * val - T(cyl_neumann(dv + 1.0, dx_v));

	T dnu;
	const double pi = std::numbers::pi;

	if (dx_v > 30.0)
	{
		dnu = -(pi * 0.5) * T(cyl_bessel_j(dv, dx_v));
	}
	else if (std::abs(dv - std::round(dv)) < 1e-10)
	{
		int n = (int)std::round(dv);
		auto res = bessel_nu_der_full(v_nu, v_x, -1);
		auto res_m = bessel_nu_der_full(-v_nu, v_x, -1);
		dnu = T(1.0 / (2.0 * pi)) * (res.d2 - T(double((n % 2 == 0) ? 1 : -1)) * res_m.d2) - T(pi * 0.5) * T(cyl_bessel_j(dv, dx_v));
	}
	else
	{
		auto res = bessel_nu_der_full(v_nu, v_x, -1);
		auto res_m = bessel_nu_der_full(-v_nu, v_x, -1);
		dnu = (res.d1 * cos(v_nu * T(pi)) - pi * T(cyl_bessel_j(dv, dx_v)) * sin(v_nu * T(pi)) + res_m.d1) / sin(v_nu * T(pi)) - T(pi) * val / tan(v_nu * T(pi));
	}

	return dual<T>(val, dnu * nu.der + dx * x.der);
}

template <typename T> inline dual<T> cyl_bessel_k(const dual<T>& nu, const dual<T>& x)
{
	using std::cyl_bessel_k; using std::sin; using std::tan;

	T v_nu = nu.val; T v_x = x.val;
	double dv = get_value(v_nu); double dx_v = get_value(v_x);
	T val = T(cyl_bessel_k(dv, dx_v));
	T dx = (v_nu / v_x) * val - T(cyl_bessel_k(dv + 1.0, dx_v));

	T dnu;
	const double pi = std::numbers::pi;

	if (dx_v > 25.0)
	{
		dnu = val * (v_nu / v_x);
	}
	else if (std::abs(dv - std::round(dv)) < 1e-10)
	{
		int n = (int)std::round(dv);
		auto res = bessel_nu_der_full(v_nu, v_x, 1);
		auto res_m = bessel_nu_der_full(-v_nu, v_x, 1);
		dnu = T(double((n % 2 == 0) ? 1 : -1) * 0.25) * (res_m.d2 - res.d2);
	}
	else
	{
		auto res = bessel_nu_der_full(v_nu, v_x, 1);
		auto res_m = bessel_nu_der_full(-v_nu, v_x, 1);
		dnu = T(pi * 0.5) * (-res_m.d1 - res.d1) / sin(v_nu * T(pi)) - T(pi) * val / tan(v_nu * T(pi));
	}

	return dual<T>(val, dnu * nu.der + dx * x.der);
}

template <typename T> inline dual<T> sph_bessel(unsigned n, const dual<T>& x)
{
	using std::sph_bessel;

	T val = sph_bessel(n, x.val);
	T der = (n == 0) ? -sph_bessel(1, x.val) : sph_bessel(n - 1, x.val) - T(double(n + 1)) / x.val * val;
	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> sph_neumann(unsigned n, const dual<T>& x)
{
	using std::sph_neumann;

	T val = sph_neumann(n, x.val);
	T der = (n == 0) ? -sph_neumann(1, x.val) : sph_neumann(n - 1, x.val) - T(double(n + 1)) / x.val * val;
	
	return dual<T>(val, der * x.der);
}

template <typename T> inline dual<T> sph_legendre(unsigned l, unsigned m, const dual<T>& theta)
{
	using std::sph_legendre; using std::sin; using std::cos; using std::sqrt;
	
	T v_t = theta.val;
	T val = T(sph_legendre(l, m, get_value(v_t)));
	
	if (l == 0)
	{
		return dual<T>(val, T(0.0));
	}

	if (m > l) 
	{
		return dual<T>(T(0.0), T(0.0));
	}

	T der;
	double vt_val = get_value(v_t);
	double sin_t = std::sin(vt_val);

	if (std::abs(sin_t) < 1e-10)
	{
		if (m == 1)
		{
			der = T(0.5) * sqrt(T(double(l * (l + 1))));
			if (vt_val > 1.0) der = -der;
		}
		else
		{
			der = T(0.0);
		}
	}
	else
	{
		T y_prev = (l > m) ? T(sph_legendre(l - 1, m, vt_val)) : T(0.0);
		T coeff = sqrt(T(double(2 * l + 1)) / T(double(2 * l - 1)) * T(double(l * l - m * m)));

		der = (T(double(l)) * cos(v_t) * val - coeff * y_prev) / sin(v_t);
	}
	
	return dual<T>(val, der * theta.der);
}

template <typename T> inline dual<T> ellint_1(const dual<T>& k, const dual<T>& phi)
{
	using std::ellint_1; using std::ellint_2; using std::sin; using std::cos; using std::sqrt;

	T vk = k.val; T vp = phi.val;
	T val = ellint_1(vk, vp);
	T k2 = vk * vk; T sn = sin(vp); T cn = cos(vp);
	T delta = sqrt(T(1.0) - k2 * sn * sn);
	T d_dk = (ellint_2(vk, vp) - (T(1.0) - k2) * val) / (vk * (T(1.0) - k2)) - (vk * sn * cn) / ((T(1.0) - k2) * delta);
	
	return dual<T>(val, d_dk * k.der + (T(1.0) / delta) * phi.der);
}

template <typename T> inline dual<T> comp_ellint_1(const dual<T>& k)
{
	using std::comp_ellint_1; using std::comp_ellint_2;

	T vk = k.val; T val = comp_ellint_1(vk);
	
	return dual<T>(val, (comp_ellint_2(vk) / (vk * (T(1.0) - vk * vk)) - val / vk) * k.der);
}

template <typename T> inline dual<T> ellint_2(const dual<T>& k, const dual<T>& phi)
{
	using std::ellint_1; using std::ellint_2; using std::sin; using std::sqrt;

	T vk = k.val; T vp = phi.val;
	T val = ellint_2(vk, vp);
	T delta = sqrt(T(1.0) - vk * vk * sin(vp) * sin(vp));
	
	return dual<T>(val, ((val - ellint_1(vk, vp)) / vk) * k.der + delta * phi.der);
}

template <typename T> inline dual<T> comp_ellint_2(const dual<T>& k)
{
	using std::comp_ellint_1; using std::comp_ellint_2;

	T vk = k.val;
	
	return dual<T>(comp_ellint_2(vk), ((comp_ellint_2(vk) - comp_ellint_1(vk)) / vk) * k.der);
}

template <typename T> inline dual<T> ellint_3(const dual<T>& nu, const dual<T>& k, const dual<T>& phi)
{
	using std::ellint_1; using std::ellint_2; using std::ellint_3;
	using std::sin; using std::cos; using std::sqrt;

	T v_n = nu.val; T v_k = k.val; T v_p = phi.val;
	// C++17/20 標準ライブラリの引数順序: (k, n, phi)
	T val = ellint_3(v_k, v_n, v_p); 

	T sn = sin(v_p); T cn = cos(v_p);
	T k2 = v_k * v_k; T n2 = v_n * v_n;
	T delta = sqrt(T(1.0) - k2 * sn * sn);

	// 1. φに関する微分 (dPi/dphi)
	T d_dp = T(1.0) / ((T(1.0) - v_n * sn * sn) * delta);
	
	// 2. kに関する微分 (dPi/dk)
	// 解析解: (k/(k^2-n)) * [ E/(1-k^2) - Pi - (k^2*sn*cn)/((1-k^2)*delta) ]
	// これにより complete 版との整合性と境界項の寄与を両立させる
	T d_dk = (v_k / (k2 - v_n)) * (
		ellint_2(v_k, v_p) / (T(1.0) - k2) - val - (k2 * sn * cn) / ((T(1.0) - k2) * delta)
	);
	
	// 3. nに関する微分 (dPi/dn)
	// 解析解: (1/(2(n-1)(k^2-n))) * [ E + (k^2-n)/n*F + (n^2-k^2)/n*Pi - (n*sn*cn*delta)/(1-n*sn^2) ]
	T d_dn = (T(1.0) / (T(2.0) * (v_n - T(1.0)) * (k2 - v_n))) * (
		ellint_2(v_k, v_p) + 
		((k2 - v_n) / v_n) * ellint_1(v_k, v_p) + 
		((n2 - k2) / v_n) * val - 
		(v_n * sn * cn * delta) / (T(1.0) - v_n * sn * sn)
	);

	return dual<T>(val, d_dn * nu.der + d_dk * k.der + d_dp * phi.der);
}

template <typename T> inline dual<T> comp_ellint_3(const dual<T>& nu, const dual<T>& k)
{
	using std::comp_ellint_1; using std::comp_ellint_2; using std::comp_ellint_3;

	T v_n = nu.val; T v_k = k.val;
	T val = comp_ellint_3(v_k, v_n);
	T k2 = v_k * v_k;

	T d_dk = (v_k / (k2 - v_n)) * (comp_ellint_2(v_k) / (T(1.0) - k2) - val);
	T d_dn = (T(1.0) / (T(2.0) * (v_n - T(1.0)) * (k2 - v_n))) * (comp_ellint_2(v_k) + ((k2 - v_n) / v_n) * comp_ellint_1(v_k) + ((v_n * v_n - k2) / v_n) * val);

	return dual<T>(val, d_dn * nu.der + d_dk * k.der);
}

template <typename T> inline dual<T> expint(const dual<T>& x)
{
	using std::expint; using std::exp;
	return dual<T>(expint(x.val), exp(x.val) / x.val * x.der);
}

template <typename T> inline dual<T> erf(const dual<T>& x)
{
	using std::sqrt; using std::exp; using std::erf;
	T val = erf(x.val);
	T der = x.der * T(2.0) / sqrt(T(std::numbers::pi)) * exp(-x.val * x.val);
	return dual<T>(val, der);
}

template <typename T> inline dual<T> erfc(const dual<T>& x)
{
	using std::sqrt; using std::exp; using std::erfc;
	T val = erfc(x.val);
	T der = -x.der * T(2.0) / sqrt(T(std::numbers::pi)) * exp(-x.val * x.val);
	return dual<T>(val, der);
}

template <typename T> inline dual<T> operator+(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) + rhs;
}

template <typename T> inline dual<T> operator-(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) - rhs;
}

template <typename T> inline dual<T> operator*(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) * rhs;
}

template <typename T> inline dual<T> operator/(const T& lhs, const dual<T>& rhs)
{
	return dual<T>(lhs) / rhs;
}

template <typename T> inline bool operator<(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val < rhs.val;
}

template <typename T> inline bool operator>(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val > rhs.val;
}

template <typename T> inline bool operator<=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val <= rhs.val;
}

template <typename T> inline bool operator>=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val >= rhs.val;
}

template <typename T> inline bool operator==(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val == rhs.val;
}

template <typename T> inline bool operator!=(const dual<T>& lhs, const dual<T>& rhs)
{
	return lhs.val != rhs.val;
}

template <typename T> inline std::ostream& operator<<(std::ostream& os, const dual<T>& d)
{
	os << "(" << d.val << ", " << d.der << ")";
	
	return os;
}

template <typename T> inline dual<T>& dual<T>::operator+=(const dual<T>& rhs)
{
	val += rhs.val; der += rhs.der;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator-=(const dual<T>& rhs)
{
	val -= rhs.val; der -= rhs.der;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator*=(const dual<T>& rhs)
{
	der = val * rhs.der + der * rhs.val; val *= rhs.val;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator/=(const dual<T>& rhs)
{
	der = (der * rhs.val - val * rhs.der) / (rhs.val * rhs.val); val /= rhs.val;
	return *this;
}

template <typename T> inline dual<T>& dual<T>::operator+=(const T& rhs)
{
	val += rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator-=(const T& rhs)
{
	val -= rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator*=(const T& rhs)
{
	val *= rhs; der *= rhs; return *this;
}

template <typename T> inline dual<T>& dual<T>::operator/=(const T& rhs)
{
	val /= rhs; der /= rhs; return *this;
}

template <typename T> inline dual<T> fmax(const dual<T>& x, const dual<T>& y)
{
	return (x.val > y.val) ? x : y;
}

template <typename T> inline dual<T> fmax(const dual<T>& x, const T& y)
{
	return (x.val > y) ? x : dual<T>(y, 0.0);
}

template <typename T> inline dual<T> fmin(const dual<T>& x, const dual<T>& y)
{
	return (x.val < y.val) ? x : y;
}

template <typename T> inline dual<T> fmin(const dual<T>& x, const T& y)
{
	return (x.val < y) ? x : dual<T>(y, 0.0);
}

template <typename T> inline bool isnan(const dual<T>& x)
{
	using std::isnan;
	return isnan(x.val);
}

template <typename T> inline bool isinf(const dual<T>& x)
{
	using std::isinf;
	return isinf(x.val);
}

template <typename T> inline dual<T> copysign(const dual<T>& x, const dual<T>& y)
{
	using std::signbit;
	using std::copysign;

	T val = copysign(x.val, y.val);
	bool same_sign = (signbit(x.val) == signbit(y.val));

	return dual<T>(val, same_sign ? x.der : -x.der);
}

template <typename T> inline dual<T> copysign(const dual<T>& x, const T& y)
{
	using std::signbit;
	using std::copysign;
	T val = copysign(x.val, y);
	bool same_sign = (signbit(x.val) == signbit(y));
	return dual<T>(val, same_sign ? x.der : -x.der);
}

template <typename T> inline dual<T> copysign(const T& x, const dual<T>& y)
{
	using std::copysign;
	return dual<T>(copysign(x, y.val), T(0.0));
}

template <typename T> inline dual<T> floor(const dual<T>& x)
{
	using std::floor;
	return dual<T>(floor(x.val), T(0.0));
}

template <typename T> inline dual<T> ceil(const dual<T>& x)
{
	using std::ceil;
	return dual<T>(ceil(x.val), T(0.0));
}

template <typename T> inline dual<T> trunc(const dual<T>& x)
{
	using std::trunc;
	return dual<T>(trunc(x.val), T(0.0));
}

template <typename T> inline dual<T> round(const dual<T>& x)
{
	using std::round;
	return dual<T>(round(x.val), T(0.0));
}

// 複素数構造体
template <typename T> struct complex
{
	T re;
	T im;

	complex(T r = T(0.0), T i = T(0.0)) : re(r), im(i)
	{
		;
	}

	template <typename U> complex(U val) : re(T(val)), im(T(0.0))
	{
		;
	}

	template <typename U> complex(const std::complex<U>& c) : re(T(c.real())), im(T(c.imag()))
	{
		;
	}

	template <typename U> explicit operator std::complex<U>() const;

	T real() const;
	T imag() const;

	complex operator+(const complex&) const;
	complex operator-(const complex&) const;
	complex operator-() const;
	complex operator*(const complex&) const;
	complex operator/(const complex&) const;

	complex& operator+=(const complex&);
	complex& operator-=(const complex&);
	complex& operator*=(const complex&);
	complex& operator/=(const complex&);

	complex& operator+=(const T&);
	complex& operator-=(const T&);
	complex& operator*=(const T&);
	complex& operator/=(const T&);
};

template <typename T> template <typename U> inline complex<T>::operator std::complex<U>() const
{
	return std::complex<U>(static_cast<U>(re), static_cast<U>(im));
}

template <typename T> inline T complex<T>::real() const
{
	return re;
}

template <typename T> inline T complex<T>::imag() const
{
	return im;
}

template <typename T> inline complex<T> complex<T>::operator+(const complex& o) const
{
	return complex(re + o.re, im + o.im);
}

template <typename T> inline complex<T> complex<T>::operator-(const complex& o) const
{
	return complex(re - o.re, im - o.im);
}

template <typename T> inline complex<T> complex<T>::operator*(const complex& o) const
{ 
	return complex(re * o.re - im * o.im, re * o.im + im * o.re); 
}

template <typename T> inline complex<T> complex<T>::operator/(const complex& o) const
{
	T denom = o.re * o.re + o.im * o.im;
	return complex((re * o.re + im * o.im) / denom, (im * o.re - re * o.im) / denom);
}

template <typename T> inline complex<T> complex<T>::operator-() const
{
	return complex(-re, -im);
}

template <typename T> std::ostream& operator<<(std::ostream& os, const complex<T>& c)
{
	os << "(" << c.re << ", " << c.im << ")";
	return os;
}

template <typename T> complex<T> operator+(const T& a, const complex<T>& b)
{
	return complex<T>(a + b.re, b.im);
}

template <typename T> complex<T> operator-(const T& a, const complex<T>& b)
{
	return complex<T>(a - b.re, -b.im);
}

template <typename T> complex<T> operator*(const T& a, const complex<T>& b)
{
	return complex<T>(a * b.re, a * b.im);
}

template <typename T> complex<T> operator/(const T& a, const complex<T>& b)
{
	T denom = b.re * b.re + b.im * b.im;
	return complex<T>(a * b.re / denom, -a * b.im / denom);
}

template <typename T> complex<T> operator+(const complex<T>& a, const T& b)
{
	return complex<T>(a.re + b, a.im);
}

template <typename T> complex<T> operator-(const complex<T>& a, const T& b)
{
	return complex<T>(a.re - b, a.im);
}

template <typename T> complex<T> operator*(const complex<T>& a, const T& b)
{
	return complex<T>(a.re * b, a.im * b);
}

template <typename T> complex<T> operator/(const complex<T>& a, const T& b)
{
	return complex<T>(a.re / b, a.im / b);
}

template <typename T> inline T abs(const complex<T>& c)
{
	using std::sqrt;
	return sqrt(c.re * c.re + c.im * c.im);
}

template <typename T> inline T norm(const complex<T>& c)
{
	return c.re * c.re + c.im * c.im;
}

template <typename T> inline complex<T> pow(const complex<T>& base, const complex<T>& exponent)
{
	// complex^complex : z^w = exp(w * log(z))
	return exp(exponent * log(base));
}

template <typename T> inline complex<T> pow(const complex<T>& base, const T& exponent)
{
	// complex^scalar
	return pow(base, complex<T>(exponent, T(0.0)));
}

template <typename T> inline complex<T> pow(const T& base, const complex<T>& exponent)
{
	// scalar^complex
	return pow(complex<T>(base, T(0.0)), exponent);
}

template <typename T> inline complex<T> conj(const complex<T>& c)
{
	return complex<T>(c.re, -c.im);
}

template <typename T> inline T arg(const complex<T>& c)
{
	using std::atan2;
	auto result = atan2(c.im, c.re);
	return result;
}

template <typename T> complex<T> sin(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;
	return complex<T>(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

template <typename T> complex<T> cos(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;
	return complex<T>(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}

template <typename T> complex<T> tan(const complex<T>& z)
{
	using std::sin; using std::cos; using std::sinh; using std::cosh;

	T sx = sin(z.re);
	T cx = cos(z.re);
	T shy = sinh(z.im);
	T chy = cosh(z.im);
	
	complex<T> sinz(sx * chy, cx * shy);
	complex<T> cosz(cx * chy, -sx * shy);
	
	return sinz / cosz;
}

template <typename T> inline complex<T> sinh(const complex<T>& z)
{
	using std::sinh; using std::cosh; using std::sin; using std::cos;
	return complex<T>(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im));
}

template <typename T> inline complex<T> cosh(const complex<T>& z)
{
	using std::sinh; using std::cosh; using std::sin; using std::cos;
	return complex<T>(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im));
}

template <typename T> inline complex<T> tanh(const complex<T>& z)
{
	return sinh(z) / cosh(z);
}

template <typename T> complex<T> exp(const complex<T>& z)
{
	using std::exp; using std::sin; using std::cos;
	T ex = exp(z.re);
	return complex<T>(ex * cos(z.im), ex * sin(z.im));
}

template <typename T> inline complex<T> log(const complex<T>& z)
{
	using std::log;
	// norm(z) = re^2 + im^2
	// arg(z) = atan2(im, re) 
	T half = T(0.5);
	return complex<T>(half * log(norm(z)), arg(z));
}

template <typename T> bool operator==(const complex<T>& a, const complex<T>& b)
{
	return (a.re == b.re) && (a.im == b.im);
}

template <typename T> bool operator!=(const complex<T>& a, const complex<T>& b)
{
	return !(a == b);
}

template <typename T> inline complex<T>& complex<T>::operator+=(const complex<T>& rhs)
{
	re += rhs.re; im += rhs.im;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator-=(const complex<T>& rhs)
{
	re -= rhs.re; im -= rhs.im;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator*=(const complex<T>& rhs)
{
	T temp_re = re * rhs.re - im * rhs.im;
	im = re * rhs.im + im * rhs.re;
	re = temp_re;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator/=(const complex<T>& rhs)
{
	T denom = rhs.re * rhs.re + rhs.im * rhs.im;
	T temp_re = (re * rhs.re + im * rhs.im) / denom;
	im = (im * rhs.re - re * rhs.im) / denom;
	re = temp_re;
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator+=(const T& rhs)
{
	re += rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator-=(const T& rhs)
{
	re -= rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator*=(const T& rhs)
{
	re *= rhs; im *= rhs; 
	return *this;
}

template <typename T> inline complex<T>& complex<T>::operator/=(const T& rhs)
{
	re /= rhs; im /= rhs; 
	return *this;
}

template <typename T> inline bool isnan(const complex<T>& c)
{
	using std::isnan;
	return isnan(c.re) || isnan(c.im);
}

template <typename T> inline T sqr(T x)
{
	return x * x;
}

template <typename T> inline T sinh_taylor(T x)
{
	T x2 = x * x;
	return x * (T(1.0) + x2 * (T(1.0 / 6.0) + x2 * T(1.0 / 120.0)));
}

template <typename T> inline T sinc(T x, T sinx)
{
	using std::abs;

	if (abs(get_value(x)) < 1.0E-4)
	{
		return T(1.0) - T(1.0 / 6.0) * x * x;
	}
	else
	{
		return sinx / x;
	}
}

template <typename T> static T erfcx_y100(T y100)
{
	switch(static_cast<int>(get_value(y100)))
	{
		case 0:
		{
			T t = T(2.0) * y100 - T(1.0);
			return T(0.70878032454106438663E-3) + (T(0.71234091047026302958E-3) + (T(0.35779077297597742384E-5) + (T(0.17403143962587937815E-7) + (T(0.81710660047307788845E-10) + (T(0.36885022360434957634E-12) + T(0.15917038551111111111E-14) * t) * t) * t) * t) * t) * t;
		}
		case 1:
		{
			T t = T(2.0) * y100 - T(3.0);
			return T(0.21479143208285144230E-2) + (T(0.72686402367379996033E-3) + (T(0.36843175430938995552E-5) + (T(0.18071841272149201685E-7) + (T(0.85496449296040325555E-10) + (T(0.38852037518534291510E-12) + T(0.16868473576888888889E-14) * t) * t) * t) * t) * t) * t;
		}
		case 2:
		{
			T t = T(2.0) * y100 - T(5.0);
			return T(0.36165255935630175090E-2) + (T(0.74182092323555510862E-3) + (T(0.37948319957528242260E-5) + (T(0.18771627021793087350E-7) + (T(0.89484715122415089123E-10) + (T(0.40935858517772440862E-12) + T(0.17872061464888888889E-14) * t) * t) * t) * t) * t) * t;
		}
		case 3:
		{
			T t = T(2.0) * y100 - T(7.0);
			return T(0.51154983860031979264E-2) + (T(0.75722840734791660540E-3) + (T(0.39096425726735703941E-5) + (T(0.19504168704300468210E-7) + (T(0.93687503063178993915E-10) + (T(0.43143925959079664747E-12) + T(0.18939926435555555556E-14) * t) * t) * t) * t) * t) * t;
		}
		case 4:
		{
			T t = T(2.0) * y100 - T(9.0);
			return T(0.66457513172673049824E-2) + (T(0.77310406054447454920E-3) + (T(0.40289510589399439385E-5) + (T(0.20271233238288381092E-7) + (T(0.98117631321709100264E-10) + (T(0.45484207406017752971E-12) + T(0.20076352213333333333E-14) * t) * t) * t) * t) * t) * t;
		}
		case 5:
		{
			T t = T(2.0) * y100 - T(11.0);
			return T(0.82082389970241207883E-2) + (T(0.78946629611881710721E-3) + (T(0.41529701552622656574E-5) + (T(0.21074693344544655714E-7) + (T(0.10278874108587317989E-9) + (T(0.47965201390613339638E-12) + T(0.21285907413333333333E-14) * t) * t) * t) * t) * t) * t;
		}
		case 6:
		{
			T t = T(2.0) * y100 - T(13.0);
			return T(0.98039537275352193165E-2) + (T(0.80633440108342840956E-3) + (T(0.42819241329736982942E-5) + (T(0.21916534346907168612E-7) + (T(0.10771535136565470914E-9) + (T(0.50595972623692822410E-12) + T(0.22573462684444444444E-14) * t) * t) * t) * t) * t) * t;
		}
		case 7:
		{
			T t = T(2.0) * y100 - T(15.0);
			return T(0.11433927298290302370E-1) + (T(0.82372858383196561209E-3) + (T(0.44160495311765438816E-5) + (T(0.22798861426211986056E-7) + (T(0.11291291745879239736E-9) + (T(0.53386189365816880454E-12) + T(0.23944209546666666667E-14) * t) * t) * t) * t) * t) * t;
		}
		case 8:
		{
			T t = T(2.0) * y100 - T(17.0);
			return T(0.13099232878814653979E-1) + (T(0.84167002467906968214E-3) + (T(0.45555958988457506002E-5) + (T(0.23723907357214175198E-7) + (T(0.11839789326602695603E-9) + (T(0.56346163067550237877E-12) + T(0.25403679644444444444E-14) * t) * t) * t) * t) * t) * t;
		}
		case 9:
		{
			T t = T(2.0) * y100 - T(19.0);
			return T(0.14800987015587535621E-1) + (T(0.86018092946345943214E-3) + (T(0.47008265848816866105E-5) + (T(0.24694040760197315333E-7) + (T(0.12418779768752299093E-9) + (T(0.59486890370320261949E-12) + T(0.26957764568888888889E-14) * t) * t) * t) * t) * t) * t;
		}
		case 10:
		{
			T t = T(2.0) * y100 - T(21.0);
			return T(0.16540351739394069380E-1) + (T(0.87928458641241463952E-3) + (T(0.48520195793001753903E-5) + (T(0.25711774900881709176E-7) + (T(0.13030128534230822419E-9) + (T(0.62820097586874779402E-12) + T(0.28612737351111111111E-14) * t) * t) * t) * t) * t) * t;
		}
		case 11:
		{
			T t = T(2.0) * y100 - T(23.0);
			return T(0.18318536789842392647E-1) + (T(0.89900542647891721692E-3) + (T(0.50094684089553365810E-5) + (T(0.26779777074218070482E-7) + (T(0.13675822186304615566E-9) + (T(0.66358287745352705725E-12) + T(0.30375273884444444444E-14) * t) * t) * t) * t) * t) * t;
		}
		case 12:
		{
			T t = T(2.0) * y100 - T(25.0);
			return T(0.20136801964214276775E-1) + (T(0.91936908737673676012E-3) + (T(0.51734830914104276820E-5) + (T(0.27900878609710432673E-7) + (T(0.14357976402809042257E-9) + (T(0.70114790311043728387E-12) + T(0.32252476000000000000E-14) * t) * t) * t) * t) * t) * t;
		}
		case 13:
		{
			T t = T(2.0) * y100 - T(27.0);
			return T(0.21996459598282740954E-1) + (T(0.94040248155366777784E-3) + (T(0.53443911508041164739E-5) + (T(0.29078085538049374673E-7) + (T(0.15078844500329731137E-9) + (T(0.74103813647499204269E-12) + T(0.34251892320000000000E-14) * t) * t) * t) * t) * t) * t;
		}
		case 14:
		{
			T t = T(2.0) * y100 - T(29.0);
			return T(0.23898877187226319502E-1) + (T(0.96213386835900177540E-3) + (T(0.55225386998049012752E-5) + (T(0.30314589961047687059E-7) + (T(0.15840826497296335264E-9) + (T(0.78340500472414454395E-12) + T(0.36381553564444444445E-14) * t) * t) * t) * t) * t) * t;
		}
		case 15:
		{
			T t = T(2.0) * y100 - T(31.0);
			return T(0.25845480155298518485E-1) + (T(0.98459293067820123389E-3) + (T(0.57082915920051843672E-5) + (T(0.31613782169164830118E-7) + (T(0.16646478745529630813E-9) + (T(0.82840985928785407942E-12) + T(0.38649975768888888890E-14) * t) * t) * t) * t) * t) * t;
		}
		case 16:
		{
			T t = T(2.0) * y100 - T(33.0);
			return T(0.27837754783474696598E-1) + (T(0.10078108563256892757E-2) + (T(0.59020366493792212221E-5) + (T(0.32979263553246520417E-7) + (T(0.17498524159268458073E-9) + (T(0.87622459124842525110E-12) + T(0.41066206488888888890E-14) * t) * t) * t) * t) * t) * t;
		}
		case 17:
		{
			T t = T(2.0) * y100 - T(35.0);
			return T(0.29877251304899307550E-1) + (T(0.10318204245057349310E-2) + (T(0.61041829697162055093E-5) + (T(0.34414860359542720579E-7) + (T(0.18399863072934089607E-9) + (T(0.92703227366365046533E-12) + T(0.43639844053333333334E-14) * t) * t) * t) * t) * t) * t;
		}
		case 18:
		{
			T t = T(2.0) * y100 - T(37.0);
			return T(0.31965587178596443475E-1) + (T(0.10566560976716574401E-2) + (T(0.63151633192414586770E-5) + (T(0.35924638339521924242E-7) + (T(0.19353584758781174038E-9) + (T(0.98102783859889264382E-12) + T(0.46381060817777777779E-14) * t) * t) * t) * t) * t) * t;
		}
		case 19:
		{
			T t = T(2.0) * y100 - T(39.0);
			return T(0.34104450552588334840E-1) + (T(0.10823541191350532574E-2) + (T(0.65354356159553934436E-5) + (T(0.37512918348533521149E-7) + (T(0.20362979635817883229E-9) + (T(0.10384187833037282363E-11) + T(0.49300625262222222221E-14) * t) * t) * t) * t) * t) * t;
		}
		case 20:
		{
			T t = T(2.0) * y100 - T(41.0);
			return T(0.36295603928292425716E-1) + (T(0.11089526167995268200E-2) + (T(0.67654845095518363577E-5) + (T(0.39184292949913591646E-7) + (T(0.21431552202133775150E-9) + (T(0.10994259106646731797E-11) + T(0.52409949102222222221E-14) * t) * t) * t) * t) * t) * t;
		}
		case 21:
		{
			T t = T(2.0) * y100 - T(43.0);
			return T(0.38540888038840509795E-1) + (T(0.11364917134175420009E-2) + (T(0.70058230641246312003E-5) + (T(0.40943644083718586939E-7) + (T(0.22563034723692881631E-9) + (T(0.11642841011361992885E-11) + T(0.55721092871111111110E-14) * t) * t) * t) * t) * t) * t;
		}
		case 22:
		{
			T t = T(2.0) * y100 - T(45.0);
			return T(0.40842225954785960651E-1) + (T(0.11650136437945673891E-2) + (T(0.72569945502343006619E-5) + (T(0.42796161861855042273E-7) + (T(0.23761401711005024162E-9) + (T(0.12332431172381557035E-11) + T(0.59246802364444444445E-14) * t) * t) * t) * t) * t) * t;
		}
		case 23:
		{
			T t = T(2.0) * y100 - T(47.0);
			return T(0.43201627431540222422E-1) + (T(0.11945628793917272199E-2) + (T(0.75195743532849206263E-5) + (T(0.44747364553960993492E-7) + (T(0.25030885216472953674E-9) + (T(0.13065684400300476484E-11) + T(0.63000532853333333334E-14) * t) * t) * t) * t) * t) * t;
		}
		case 24:
		{
			T t = T(2.0) * y100 - T(49.0);
			return T(0.45621193513810471438E-1) + (T(0.12251862608067529503E-2) + (T(0.77941720055551920319E-5) + (T(0.46803119830954460212E-7) + (T(0.26375990983978426273E-9) + (T(0.13845421370977119765E-11) + T(0.66996477404444444445E-14) * t) * t) * t) * t) * t) * t;
		}
		case 25:
		{
			T t = T(2.0) * y100 - T(51.0);
			return T(0.48103121413299865517E-1) + (T(0.12569331386432195113E-2) + (T(0.80814333496367673980E-5) + (T(0.48969667335682018324E-7) + (T(0.27801515481905748484E-9) + (T(0.14674637611609884208E-11) + T(0.71249589351111111110E-14) * t) * t) * t) * t) * t) * t;
		}
		case 26:
		{
			T t = T(2.0) * y100 - T(53.0);
			return T(0.50649709676983338501E-1) + (T(0.12898555233099055810E-2) + (T(0.83820428414568799654E-5) + (T(0.51253642652551838659E-7) + (T(0.29312563849675507232E-9) + (T(0.15556512782814827846E-11) + T(0.75775607822222222221E-14) * t) * t) * t) * t) * t) * t;
		}
		case 27:
		{
			T t = T(2.0) * y100 - T(55.0);
			return T(0.53263363664388864181E-1) + (T(0.13240082443256975769E-2) + (T(0.86967260015007658418E-5) + (T(0.53662102750396795566E-7) + (T(0.30914568786634796807E-9) + (T(0.16494420240828493176E-11) + T(0.80591079644444444445E-14) * t) * t) * t) * t) * t) * t;
		}
		case 28:
		{
			T t = T(2.0) * y100 - T(57.0);
			return T(0.55946601353500013794E-1) + (T(0.13594491197408190706E-2) + (T(0.90262520233016380987E-5) + (T(0.56202552975056695376E-7) + (T(0.32613310410503135996E-9) + (T(0.17491936862246367398E-11) + T(0.85713381688888888890E-14) * t) * t) * t) * t) * t) * t;
		}
		case 29:
		{
			T t = T(2.0) * y100 - T(59.0);
			return T(0.58702059496154081813E-1) + (T(0.13962391363223647892E-2) + (T(0.93714365487312784270E-5) + (T(0.58882975670265286526E-7) + (T(0.34414937110591753387E-9) + (T(0.18552853109751857859E-11) + T(0.91160736711111111110E-14) * t) * t) * t) * t) * t) * t;
		}
		case 30:
		{
			T t = T(2.0) * y100 - T(61.0);
			return T(0.61532500145144778048E-1) + (T(0.14344426411912015247E-2) + (T(0.97331446201016809696E-5) + (T(0.61711860507347175097E-7) + (T(0.36325987418295300221E-9) + (T(0.19681183310134518232E-11) + T(0.96952238400000000000E-14) * t) * t) * t) * t) * t) * t;
		}
		case 31:
		{
			T t = T(2.0) * y100 - T(63.0);
			return T(0.64440817576653297993E-1) + (T(0.14741275456383131151E-2) + (T(0.10112293819576437838E-4) + (T(0.64698236605933246196E-7) + (T(0.38353412915303665586E-9) + (T(0.20881176114385120186E-11) + T(0.10310784480000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 32:
		{
			T t = T(2.0) * y100 - T(65.0);
			return T(0.67430045633130393282E-1) + (T(0.15153655418916540370E-2) + (T(0.10509857606888328667E-4) + (T(0.67851706529363332855E-7) + (T(0.40504602194811140006E-9) + (T(0.22157325110542534469E-11) + T(0.10964842115555555556E-13) * t) * t) * t) * t) * t) * t;
		}
		case 33:
		{
			T t = T(2.0) * y100 - T(67.0);
			return T(0.70503365513338850709E-1) + (T(0.15582323336495709827E-2) + (T(0.10926868866865231089E-4) + (T(0.71182482239613507542E-7) + (T(0.42787405890153386710E-9) + (T(0.23514379522274416437E-11) + T(0.11659571751111111111E-13) * t) * t) * t) * t) * t) * t;
		}
		case 34:
		{
			T t = T(2.0) * y100 - T(69.0);
			return T(0.73664114037944596353E-1) + (T(0.16028078812438820413E-2) + (T(0.11364423678778207991E-4) + (T(0.74701423097423182009E-7) + (T(0.45210162777476488324E-9) + (T(0.24957355004088569134E-11) + T(0.12397238257777777778E-13) * t) * t) * t) * t) * t) * t;
		}
		case 35:
		{
			T t = T(2.0) * y100 - T(71.0);
			return T(0.76915792420819562379E-1) + (T(0.16491766623447889354E-2) + (T(0.11823685320041302169E-4) + (T(0.78420075993781544386E-7) + (T(0.47781726956916478925E-9) + (T(0.26491544403815724749E-11) + T(0.13180196462222222222E-13) * t) * t) * t) * t) * t) * t;
		}
		case 36:
		{
			T t = T(2.0) * y100 - T(73.0);
			return T(0.80262075578094612819E-1) + (T(0.16974279491709504117E-2) + (T(0.12305888517309891674E-4) + (T(0.82350717698979042290E-7) + (T(0.50511496109857113929E-9) + (T(0.28122528497626897696E-11) + T(0.14010889635555555556E-13) * t) * t) * t) * t) * t) * t;
		}
		case 37:
		{
			T t = T(2.0) * y100 - T(75.0);
			return T(0.83706822008980357446E-1) + (T(0.17476561032212656962E-2) + (T(0.12812343958540763368E-4) + (T(0.86506399515036435592E-7) + (T(0.53409440823869467453E-9) + (T(0.29856186620887555043E-11) + T(0.14891851591111111111E-13) * t) * t) * t) * t) * t) * t;
		}
		case 38:
		{
			T t = T(2.0) * y100 - T(77.0);
			return T(0.87254084284461718231E-1) + (T(0.17999608886001962327E-2) + (T(0.13344443080089492218E-4) + (T(0.90900994316429008631E-7) + (T(0.56486134972616465316E-9) + (T(0.31698707080033956934E-11) + T(0.15825697795555555556E-13) * t) * t) * t) * t) * t) * t;
		}
		case 39:
		{
			T t = T(2.0) * y100 - T(79.0);
			return T(0.90908120182172748487E-1) + (T(0.18544478050657699758E-2) + (T(0.13903663143426120077E-4) + (T(0.95549246062549906177E-7) + (T(0.59752787125242054315E-9) + (T(0.33656597366099099413E-11) + T(0.16815130613333333333E-13) * t) * t) * t) * t) * t) * t;
		}
		case 40:
		{
			T t = T(2.0) * y100 - T(81.0);
			return T(0.94673404508075481121E-1) + (T(0.19112284419887303347E-2) + (T(0.14491572616545004930E-4) + (T(0.10046682186333613697E-6) + (T(0.63221272959791000515E-9) + (T(0.35736693975589130818E-11) + T(0.17862931591111111111E-13) * t) * t) * t) * t) * t) * t;
		}
		case 41:
		{
			T t = T(2.0) * y100 - T(83.0);
			return T(0.98554641648004456555E-1) + (T(0.19704208544725622126E-2) + (T(0.15109836875625443935E-4) + (T(0.10567036667675984067E-6) + (T(0.66904168640019354565E-9) + (T(0.37946171850824333014E-11) + T(0.18971959040000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 42:
		{
			T t = T(2.0) * y100 - T(85.0);
			return T(0.10255677889470089531E0) + (T(0.20321499629472857418E-2) + (T(0.15760224242962179564E-4) + (T(0.11117756071353507391E-6) + (T(0.70814785110097658502E-9) + (T(0.40292553276632563925E-11) + T(0.20145143075555555556E-13) * t) * t) * t) * t) * t) * t;
		}
		case 43:
		{
			T t = T(2.0) * y100 - T(87.0);
			return T(0.10668502059865093318E0) + (T(0.20965479776148731610E-2) + (T(0.16444612377624983565E-4) + (T(0.11700717962026152749E-6) + (T(0.74967203250938418991E-9) + (T(0.42783716186085922176E-11) + T(0.21385479360000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 44:
		{
			T t = T(2.0) * y100 - T(89.0);
			return T(0.11094484319386444474E0) + (T(0.21637548491908170841E-2) + (T(0.17164995035719657111E-4) + (T(0.12317915750735938089E-6) + (T(0.79376309831499633734E-9) + (T(0.45427901763106353914E-11) + T(0.22696025653333333333E-13) * t) * t) * t) * t) * t) * t;
		}
		case 45:
		{
			T t = T(2.0) * y100 - T(91.0);
			return T(0.11534201115268804714E0) + (T(0.22339187474546420375E-2) + (T(0.17923489217504226813E-4) + (T(0.12971465288245997681E-6) + (T(0.84057834180389073587E-9) + (T(0.48233721206418027227E-11) + T(0.24079890062222222222E-13) * t) * t) * t) * t) * t) * t;
		}
		case 46:
		{
			T t = T(2.0) * y100 - T(93.0);
			return T(0.11988259392684094740E0) + (T(0.23071965691918689601E-2) + (T(0.18722342718958935446E-4) + (T(0.13663611754337957520E-6) + (T(0.89028385488493287005E-9) + (T(0.51210161569225846701E-11) + T(0.25540227111111111111E-13) * t) * t) * t) * t) * t) * t;
		}
		case 47:
		{
			T t = T(2.0) * y100 - T(95.0);
			return T(0.12457298393509812907E0) + (T(0.23837544771809575380E-2) + (T(0.19563942105711612475E-4) + (T(0.14396736847739470782E-6) + (T(0.94305490646459247016E-9) + (T(0.54366590583134218096E-11) + T(0.27080225920000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 48:
		{
			T t = T(2.0) * y100 - T(97.0);
			return T(0.12941991566142438816E0) + (T(0.24637684719508859484E-2) + (T(0.20450821127475879816E-4) + (T(0.15173366280523906622E-6) + (T(0.99907632506389027739E-9) + (T(0.57712760311351625221E-11) + T(0.28703099555555555556E-13) * t) * t) * t) * t) * t) * t;
		}
		case 49:
		{
			T t = T(2.0) * y100 - T(99.0);
			return T(0.13443048593088696613E0) + (T(0.25474249981080823877E-2) + (T(0.21385669591362915223E-4) + (T(0.15996177579900443030E-6) + (T(0.10585428844575134013E-8) + (T(0.61258809536787882989E-11) + T(0.30412080142222222222E-13) * t) * t) * t) * t) * t) * t;
		}
		case 50:
		{
			T t = T(2.0) * y100 - T(101.0);
			return T(0.13961217543434561353E0) + (T(0.26349215871051761416E-2) + (T(0.22371342712572567744E-4) + (T(0.16868008199296822247E-6) + (T(0.11216596910444996246E-8) + (T(0.65015264753090890662E-11) + T(0.32210394506666666666E-13) * t) * t) * t) * t) * t) * t;
		}
		case 51:
		{
			T t = T(2.0) * y100 - T(103.0);
			return T(0.14497287157673800690E0) + (T(0.27264675383982439814E-2) + (T(0.23410870961050950197E-4) + (T(0.17791863939526376477E-6) + (T(0.11886425714330958106E-8) + (T(0.68993039665054288034E-11) + T(0.34101266222222222221E-13) * t) * t) * t) * t) * t) * t;
		}
		case 52:
		{
			T t = T(2.0) * y100 - T(105.0);
			return T(0.15052089272774618151E0) + (T(0.28222846410136238008E-2) + (T(0.24507470422713397006E-4) + (T(0.18770927679626136909E-6) + (T(0.12597184587583370712E-8) + (T(0.73203433049229821618E-11) + T(0.36087889048888888890E-13) * t) * t) * t) * t) * t) * t;
		}
		case 53:
		{
			T t = T(2.0) * y100 - T(107.0);
			return T(0.15626501395774612325E0) + (T(0.29226079376196624949E-2) + (T(0.25664553693768450545E-4) + (T(0.19808568415654461964E-6) + (T(0.13351257759815557897E-8) + (T(0.77658124891046760667E-11) + T(0.38173420035555555555E-13) * t) * t) * t) * t) * t) * t;
		}
		case 54:
		{
			T t = T(2.0) * y100 - T(109.0);
			return T(0.16221449434620737567E0) + (T(0.30276865332726475672E-2) + (T(0.26885741326534564336E-4) + (T(0.20908350604346384143E-6) + (T(0.14151148144240728728E-8) + (T(0.82369170665974313027E-11) + T(0.40360957457777777779E-13) * t) * t) * t) * t) * t) * t;
		}
		case 55:
		{
			T t = T(2.0) * y100 - T(111.0);
			return T(0.16837910595412130659E0) + (T(0.31377844510793082301E-2) + (T(0.28174873844911175026E-4) + (T(0.22074043807045782387E-6) + (T(0.14999481055996090039E-8) + (T(0.87348993661930809254E-11) + T(0.42653528977777777779E-13) * t) * t) * t) * t) * t) * t;
		}
		case 56:
		{
			T t = T(2.0) * y100 - T(113.0);
			return T(0.17476916455659369953E0) + (T(0.32531815370903068316E-2) + (T(0.29536024347344364074E-4) + (T(0.23309632627767074202E-6) + (T(0.15899007843582444846E-8) + (T(0.92610375235427359475E-11) + T(0.45054073102222222221E-13) * t) * t) * t) * t) * t) * t;
		}
		case 57:
		{
			T t = T(2.0) * y100 - T(115.0);
			return T(0.18139556223643701364E0) + (T(0.33741744168096996041E-2) + (T(0.30973511714709500836E-4) + (T(0.24619326937592290996E-6) + (T(0.16852609412267750744E-8) + (T(0.98166442942854895573E-11) + T(0.47565418097777777779E-13) * t) * t) * t) * t) * t) * t;
		}
		case 58:
		{
			T t = T(2.0) * y100 - T(117.0);
			return T(0.18826980194443664549E0) + (T(0.35010775057740317997E-2) + (T(0.32491914440014267480E-4) + (T(0.26007572375886319028E-6) + (T(0.17863299617388376116E-8) + (T(0.10403065638343878679E-10) + T(0.50190265831111111110E-13) * t) * t) * t) * t) * t) * t;
		}
		case 59:
		{
			T t = T(2.0) * y100 - T(119.0);
			return T(0.19540403413693967350E0) + (T(0.36342240767211326315E-2) + (T(0.34096085096200907289E-4) + (T(0.27479061117017637474E-6) + (T(0.18934228504790032826E-8) + (T(0.11021679075323598664E-10) + T(0.52931171733333333334E-13) * t) * t) * t) * t) * t) * t;
		}
		case 60:
		{
			T t = T(2.0) * y100 - T(121.0);
			return T(0.20281109560651886959E0) + (T(0.37739673859323597060E-2) + (T(0.35791165457592409054E-4) + (T(0.29038742889416172404E-6) + (T(0.20068685374849001770E-8) + (T(0.11673891799578381999E-10) + T(0.55790523093333333334E-13) * t) * t) * t) * t) * t) * t;
		}
		case 61:
		{
			T t = T(2.0) * y100 - T(123.0);
			return T(0.21050455062669334978E0) + (T(0.39206818613925652425E-2) + (T(0.37582602289680101704E-4) + (T(0.30691836231886877385E-6) + (T(0.21270101645763677824E-8) + (T(0.12361138551062899455E-10) + T(0.58770520160000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 62:
		{
			T t = T(2.0) * y100 - T(125.0);
			return T(0.21849873453703332479E0) + (T(0.40747643554689586041E-2) + (T(0.39476163820986711501E-4) + (T(0.32443839970139918836E-6) + (T(0.22542053491518680200E-8) + (T(0.13084879235290858490E-10) + T(0.61873153262222222221E-13) * t) * t) * t) * t) * t) * t;
		}
		case 63:
		{
			T t = T(2.0) * y100 - T(127.0);
			return T(0.22680879990043229327E0) + (T(0.42366354648628516935E-2) + (T(0.41477956909656896779E-4) + (T(0.34300544894502810002E-6) + (T(0.23888264229264067658E-8) + (T(0.13846596292818514601E-10) + T(0.65100183751111111110E-13) * t) * t) * t) * t) * t) * t;
		}
		case 64:
		{
			T t = T(2.0) * y100 - T(129.0);
			return T(0.23545076536988703937E0) + (T(0.44067409206365170888E-2) + (T(0.43594444916224700881E-4) + (T(0.36268045617760415178E-6) + (T(0.25312606430853202748E-8) + (T(0.14647791812837903061E-10) + T(0.68453122631111111110E-13) * t) * t) * t) * t) * t) * t;
		}
		case 65:
		{
			T t = T(2.0) * y100 - T(131.0);
			return T(0.24444156740777432838E0) + (T(0.45855530511605787178E-2) + (T(0.45832466292683085475E-4) + (T(0.38352752590033030472E-6) + (T(0.26819103733055603460E-8) + (T(0.15489984390884756993E-10) + T(0.71933206364444444445E-13) * t) * t) * t) * t) * t) * t;
		}
		case 66:
		{
			T t = T(2.0) * y100 - T(133.0);
			return T(0.25379911500634264643E0) + (T(0.47735723208650032167E-2) + (T(0.48199253896534185372E-4) + (T(0.40561404245564732314E-6) + (T(0.28411932320871165585E-8) + (T(0.16374705736458320149E-10) + T(0.75541379822222222221E-13) * t) * t) * t) * t) * t) * t;
		}
		case 67:
		{
			T t = T(2.0) * y100 - T(135.0);
			return T(0.26354234756393613032E0) + (T(0.49713289477083781266E-2) + (T(0.50702455036930367504E-4) + (T(0.42901079254268185722E-6) + (T(0.30095422058900481753E-8) + (T(0.17303497025347342498E-10) + T(0.79278273368888888890E-13) * t) * t) * t) * t) * t) * t;
		}
		case 68:
		{
			T t = T(2.0) * y100 - T(137.0);
			return T(0.27369129607732343398E0) + (T(0.51793846023052643767E-2) + (T(0.53350152258326602629E-4) + (T(0.45379208848865015485E-6) + (T(0.31874057245814381257E-8) + (T(0.18277905010245111046E-10) + T(0.83144182364444444445E-13) * t) * t) * t) * t) * t) * t;
		}
		case 69:
		{
			T t = T(2.0) * y100 - T(139.0);
			return T(0.28426714781640316172E0) + (T(0.53983341916695141966E-2) + (T(0.56150884865255810638E-4) + (T(0.48003589196494734238E-6) + (T(0.33752476967570796349E-8) + (T(0.19299477888083469086E-10) + T(0.87139049137777777779E-13) * t) * t) * t) * t) * t) * t;
		}
		case 70:
		{
			T t = T(2.0) * y100 - T(141.0);
			return T(0.29529231465348519920E0) + (T(0.56288077305420795663E-2) + (T(0.59113671189913307427E-4) + (T(0.50782393781744840482E-6) + (T(0.35735475025851713168E-8) + (T(0.20369760937017070382E-10) + T(0.91262442613333333334E-13) * t) * t) * t) * t) * t) * t;
		}
		case 71:
		{
			T t = T(2.0) * y100 - T(143.0);
			return T(0.30679050522528838613E0) + (T(0.58714723032745403331E-2) + (T(0.62248031602197686791E-4) + (T(0.53724185766200945789E-6) + (T(0.37827999418960232678E-8) + (T(0.21490291930444538307E-10) + T(0.95513539182222222221E-13) * t) * t) * t) * t) * t) * t;
		}
		case 72:
		{
			T t = T(2.0) * y100 - T(145.0);
			return T(0.31878680111173319425E0) + (T(0.61270341192339103514E-2) + (T(0.65564012259707640976E-4) + (T(0.56837930287837738996E-6) + (T(0.40035151353392378882E-8) + (T(0.22662596341239294792E-10) + T(0.99891109760000000000E-13) * t) * t) * t) * t) * t) * t;
		}
		case 73:
		{
			T t = T(2.0) * y100 - T(147.0);
			return T(0.33130773722152622027E0) + (T(0.63962406646798080903E-2) + (T(0.69072209592942396666E-4) + (T(0.60133006661885941812E-6) + (T(0.42362183765883466691E-8) + (T(0.23888182347073698382E-10) + T(0.10439349811555555556E-12) * t) * t) * t) * t) * t) * t;
		}
		case 74:
		{
			T t = T(2.0) * y100 - T(149.0);
			return T(0.34438138658041336523E0) + (T(0.66798829540414007258E-2) + (T(0.72783795518603561144E-4) + (T(0.63619220443228800680E-6) + (T(0.44814499336514453364E-8) + (T(0.25168535651285475274E-10) + T(0.10901861383111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 75:
		{
			T t = T(2.0) * y100 - T(151.0);
			return T(0.35803744972380175583E0) + (T(0.69787978834882685031E-2) + (T(0.76710543371454822497E-4) + (T(0.67306815308917386747E-6) + (T(0.47397647975845228205E-8) + (T(0.26505114141143050509E-10) + T(0.11376390933333333333E-12) * t) * t) * t) * t) * t) * t;
		}
		case 76:
		{
			T t = T(2.0) * y100 - T(153.0);
			return T(0.37230734890119724188E0) + (T(0.72938706896461381003E-2) + (T(0.80864854542670714092E-4) + (T(0.71206484718062688779E-6) + (T(0.50117323769745883805E-8) + (T(0.27899342394100074165E-10) + T(0.11862637614222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 77:
		{
			T t = T(2.0) * y100 - T(155.0);
			return T(0.38722432730555448223E0) + (T(0.76260375162549802745E-2) + (T(0.85259785810004603848E-4) + (T(0.75329383305171327677E-6) + (T(0.52979361368388119355E-8) + (T(0.29352606054164086709E-10) + T(0.12360253370666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 78:
		{
			T t = T(2.0) * y100 - T(157.0);
			return T(0.40282355354616940667E0) + (T(0.79762880915029728079E-2) + (T(0.89909077342438246452E-4) + (T(0.79687137961956194579E-6) + (T(0.55989731807360403195E-8) + (T(0.30866246101464869050E-10) + T(0.12868841946666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 79:
		{
			T t = T(2.0) * y100 - T(159.0);
			return T(0.41914223158913787649E0) + (T(0.83456685186950463538E-2) + (T(0.94827181359250161335E-4) + (T(0.84291858561783141014E-6) + (T(0.59154537751083485684E-8) + (T(0.32441553034347469291E-10) + T(0.13387957943111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 80:
		{
			T t = T(2.0) * y100 - T(161.0);
			return T(0.43621971639463786896E0) + (T(0.87352841828289495773E-2) + (T(0.10002929142066799966E-3) + (T(0.89156148280219880024E-6) + (T(0.62480008150788597147E-8) + (T(0.34079760983458878910E-10) + T(0.13917107176888888889E-12) * t) * t) * t) * t) * t) * t;
		}
		case 81:
		{
			T t = T(2.0) * y100 - T(163.0);
			return T(0.45409763548534330981E0) + (T(0.91463027755548240654E-2) + (T(0.10553137232446167258E-3) + (T(0.94293113464638623798E-6) + (T(0.65972492312219959885E-8) + (T(0.35782041795476563662E-10) + T(0.14455745872000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 82:
		{
			T t = T(2.0) * y100 - T(165.0);
			return T(0.47282001668512331468E0) + (T(0.95799574408860463394E-2) + (T(0.11135019058000067469E-3) + (T(0.99716373005509038080E-6) + (T(0.69638453369956970347E-8) + (T(0.37549499088161345850E-10) + T(0.15003280712888888889E-12) * t) * t) * t) * t) * t) * t;
		}
		case 83:
		{
			T t = T(2.0) * y100 - T(167.0);
			return T(0.49243342227179841649E0) + (T(0.10037550043909497071E-1) + (T(0.11750334542845234952E-3) + (T(0.10544006716188967172E-5) + (T(0.73484461168242224872E-8) + (T(0.39383162326435752965E-10) + T(0.15559069118222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 84:
		{
			T t = T(2.0) * y100 - T(169.0);
			return T(0.51298708979209258326E0) + (T(0.10520454564612427224E-1) + (T(0.12400930037494996655E-3) + (T(0.11147886579371265246E-5) + (T(0.77517184550568711454E-8) + (T(0.41283980931872622611E-10) + T(0.16122419680000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 85:
		{
			T t = T(2.0) * y100 - T(171.0);
			return T(0.53453307979101369843E0) + (T(0.11030120618800726938E-1) + (T(0.13088741519572269581E-3) + (T(0.11784797595374515432E-5) + (T(0.81743383063044825400E-8) + (T(0.43252818449517081051E-10) + T(0.16692592640000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 86:
		{
			T t = T(2.0) * y100 - T(173.0);
			return T(0.55712643071169299478E0) + (T(0.11568077107929735233E-1) + (T(0.13815797838036651289E-3) + (T(0.12456314879260904558E-5) + (T(0.86169898078969313597E-8) + (T(0.45290446811539652525E-10) + T(0.17268801084444444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 87:
		{
			T t = T(2.0) * y100 - T(175.0);
			return T(0.58082532122519320968E0) + (T(0.12135935999503877077E-1) + (T(0.14584223996665838559E-3) + (T(0.13164068573095710742E-5) + (T(0.90803643355106020163E-8) + (T(0.47397540713124619155E-10) + T(0.17850211608888888889E-12) * t) * t) * t) * t) * t) * t;
		}
		case 88:
		{
			T t = T(2.0) * y100 - T(177.0);
			return T(0.60569124025293375554E0) + (T(0.12735396239525550361E-1) + (T(0.15396244472258863344E-3) + (T(0.13909744385382818253E-5) + (T(0.95651595032306228245E-8) + (T(0.49574672127669041550E-10) + T(0.18435945564444444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 89:
		{
			T t = T(2.0) * y100 - T(179.0);
			return T(0.63178916494715716894E0) + (T(0.13368247798287030927E-1) + (T(0.16254186562762076141E-3) + (T(0.14695084048334056083E-5) + (T(0.10072078109604152350E-7) + (T(0.51822304995680707483E-10) + T(0.19025081422222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 90:
		{
			T t = T(2.0) * y100 - T(181.0);
			return T(0.65918774689725319200E0) + (T(0.14036375850601992063E-1) + (T(0.17160483760259706354E-3) + (T(0.15521885688723188371E-5) + (T(0.10601827031535280590E-7) + (T(0.54140790105837520499E-10) + T(0.19616655146666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 91:
		{
			T t = T(2.0) * y100 - T(183.0);
			return T(0.68795950683174433822E0) + (T(0.14741765091365869084E-1) + (T(0.18117679143520433835E-3) + (T(0.16392004108230585213E-5) + (T(0.11155116068018043001E-7) + (T(0.56530360194925690374E-10) + T(0.20209663662222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 92:
		{
			T t = T(2.0) * y100 - T(185.0);
			return T(0.71818103808729967036E0) + (T(0.15486504187117112279E-1) + (T(0.19128428784550923217E-3) + (T(0.17307350969359975848E-5) + (T(0.11732656736113607751E-7) + (T(0.58991125287563833603E-10) + T(0.20803065333333333333E-12) * t) * t) * t) * t) * t) * t;
		}
		case 93:
		{
			T t = T(2.0) * y100 - T(187.0);
			return T(0.74993321911726254661E0) + (T(0.16272790364044783382E-1) + (T(0.20195505163377912645E-3) + (T(0.18269894883203346953E-5) + (T(0.12335161021630225535E-7) + (T(0.61523068312169087227E-10) + T(0.21395783431111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 94:
		{
			T t = T(2.0) * y100 - T(189.0);
			return T(0.78330143531283492729E0) + (T(0.17102934132652429240E-1) + (T(0.21321800585063327041E-3) + (T(0.19281661395543913713E-5) + (T(0.12963340087354341574E-7) + (T(0.64126040998066348872E-10) + T(0.21986708942222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 95:
		{
			T t = T(2.0) * y100 - T(191.0);
			return T(0.81837581041023811832E0) + (T(0.17979364149044223802E-1) + (T(0.22510330592753129006E-3) + (T(0.20344732868018175389E-5) + (T(0.13617902941839949718E-7) + (T(0.66799760083972474642E-10) + T(0.22574701262222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 96:
		{
			T t = T(2.0) * y100 - T(193.0);
			return T(0.85525144775685126237E0) + (T(0.18904632212547561026E-1) + (T(0.23764237370371255638E-3) + (T(0.21461248251306387979E-5) + (T(0.14299555071870523786E-7) + (T(0.69543803864694171934E-10) + T(0.23158593688888888889E-12) * t) * t) * t) * t) * t) * t;
		}
		case 97:
		{
			T t = T(2.0) * y100 - T(195.0);
			return T(0.89402868170849933734E0) + (T(0.19881418399127202569E-1) + (T(0.25086793128395995798E-3) + (T(0.22633402747585233180E-5) + (T(0.15008997042116532283E-7) + (T(0.72357609075043941261E-10) + T(0.23737194737777777778E-12) * t) * t) * t) * t) * t) * t;
		}
		case 98:
		{
			T t = T(2.0) * y100 - T(197.0);
			return T(0.93481333942870796363E0) + (T(0.20912536329780368893E-1) + (T(0.26481403465998477969E-3) + (T(0.23863447359754921676E-5) + (T(0.15746923065472184451E-7) + (T(0.75240468141720143653E-10) + T(0.24309291271111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 99:
		{
			T t = T(2.0) * y100 - T(199.0);
			return T(0.97771701335885035464E0) + (T(0.22000938572830479551E-1) + (T(0.27951610702682383001E-3) + (T(0.25153688325245314530E-5) + (T(0.16514019547822821453E-7) + (T(0.78191526829368231251E-10) + T(0.24873652355555555556E-12) * t) * t) * t) * t) * t) * t;
		}
	}

	return T(1.0);
}

template <typename T> static T w_im_y100(T y100, T x)
{
	switch (static_cast<int>(get_value(y100)))
	{
		case 0:
		{
			T t = T(2.0) * y100 - T(1.0);
			return T(0.28351593328822191546E-2) + (T(0.28494783221378400759E-2) + (T(0.14427470563276734183E-4) + (T(0.10939723080231588129E-6) + (T(0.92474307943275042045E-9) + (T(0.89128907666450075245E-11) + T(0.92974121935111111110E-13) * t) * t) * t) * t) * t) * t;
		}
		case 1:
		{
			T t = T(2.0) * y100 - T(3.0);
			return T(0.85927161243940350562E-2) + (T(0.29085312941641339862E-2) + (T(0.15106783707725582090E-4) + (T(0.11716709978531327367E-6) + (T(0.10197387816021040024E-8) + (T(0.10122678863073360769E-10) + T(0.10917479678400000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 2:
		{
			T t = T(2.0) * y100 - T(5.0);
			return T(0.14471159831187703054E-1) + (T(0.29703978970263836210E-2) + (T(0.15835096760173030976E-4) + (T(0.12574803383199211596E-6) + (T(0.11278672159518415848E-8) + (T(0.11547462300333495797E-10) + T(0.12894535335111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 3:
		{
			T t = T(2.0) * y100 - T(7.0);
			return T(0.20476320420324610618E-1) + (T(0.30352843012898665856E-2) + (T(0.16617609387003727409E-4) + (T(0.13525429711163116103E-6) + (T(0.12515095552507169013E-8) + (T(0.13235687543603382345E-10) + T(0.15326595042666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 4:
		{
			T t = T(2.0) * y100 - T(9.0);
			return T(0.26614461952489004566E-1) + (T(0.31034189276234947088E-2) + (T(0.17460268109986214274E-4) + (T(0.14582130824485709573E-6) + (T(0.13935959083809746345E-8) + (T(0.15249438072998932900E-10) + T(0.18344741882133333333E-12) * t) * t) * t) * t) * t) * t;
		}
		case 5:
		{
			T t = T(2.0) * y100 - T(11.0);
			return T(0.32892330248093586215E-1) + (T(0.31750557067975068584E-2) + (T(0.18369907582308672632E-4) + (T(0.15761063702089457882E-6) + (T(0.15577638230480894382E-8) + (T(0.17663868462699097951E-10) + (T(0.22126732680711111111E-12) + T(0.30273474177737853668E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 6:
		{
			T t = T(2.0) * y100 - T(13.0);
			return T(0.39317207681134336024E-1) + (T(0.32504779701937539333E-2) + (T(0.19354426046513400534E-4) + (T(0.17081646971321290539E-6) + (T(0.17485733959327106250E-8) + (T(0.20593687304921961410E-10) + (T(0.26917401949155555556E-12) + T(0.38562123837725712270E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 7:
		{
			T t = T(2.0) * y100 - T(15.0);
			return T(0.45896976511367738235E-1) + (T(0.33300031273110976165E-2) + (T(0.20423005398039037313E-4) + (T(0.18567412470376467303E-6) + (T(0.19718038363586588213E-8) + (T(0.24175006536781219807E-10) + (T(0.33059982791466666666E-12) + T(0.49756574284439426165E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 8:
		{
			T t = T(2.0) * y100 - T(17.0);
			return T(0.52640192524848962855E-1) + (T(0.34139883358846720806E-2) + (T(0.21586390240603337337E-4) + (T(0.20247136501568904646E-6) + (T(0.22348696948197102935E-8) + (T(0.28597516301950162548E-10) + (T(0.41045502119111111110E-12) + T(0.65151614515238361946E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 9:
		{
			T t = T(2.0) * y100 - T(19.0);
			return T(0.59556171228656770456E-1) + (T(0.35028374386648914444E-2) + (T(0.22857246150998562824E-4) + (T(0.22156372146525190679E-6) + (T(0.25474171590893813583E-8) + (T(0.34122390890697400584E-10) + (T(0.51593189879111111110E-12) + T(0.86775076853908006938E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 10:
		{
			T t = T(2.0) * y100 - T(21.0);
			return T(0.66655089485108212551E-1) + (T(0.35970095381271285568E-2) + (T(0.24250626164318672928E-4) + (T(0.24339561521785040536E-6) + (T(0.29221990406518411415E-8) + (T(0.41117013527967776467E-10) + (T(0.65786450716444444445E-12) + T(0.11791885745450623331E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 11:
		{
			T t = T(2.0) * y100 - T(23.0);
			return T(0.73948106345519174661E-1) + (T(0.36970297216569341748E-2) + (T(0.25784588137312868792E-4) + (T(0.26853012002366752770E-6) + (T(0.33763958861206729592E-8) + (T(0.50111549981376976397E-10) + (T(0.85313857496888888890E-12) + T(0.16417079927706899860E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 12:
		{
			T t = T(2.0) * y100 - T(25.0);
			return T(0.81447508065002963203E-1) + (T(0.38035026606492705117E-2) + (T(0.27481027572231851896E-4) + (T(0.29769200731832331364E-6) + (T(0.39336816287457655076E-8) + (T(0.61895471132038157624E-10) + (T(0.11292303213511111111E-11) + T(0.23558532213703884304E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 13:
		{
			T t = T(2.0) * y100 - T(27.0);
			return T(0.89166884027582716628E-1) + (T(0.39171301322438946014E-2) + (T(0.29366827260422311668E-4) + (T(0.33183204390350724895E-6) + (T(0.46276006281647330524E-8) + (T(0.77692631378169813324E-10) + (T(0.15335153258844444444E-11) + T(0.35183103415916026911E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 14:
		{
			T t = T(2.0) * y100 - T(29.0);
			return T(0.97121342888032322019E-1) + (T(0.40387340353207909514E-2) + (T(0.31475490395950776930E-4) + (T(0.37222714227125135042E-6) + (T(0.55074373178613809996E-8) + (T(0.99509175283990337944E-10) + (T(0.21552645758222222222E-11) + T(0.55728651431872687605E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 15:
		{
			T t = T(2.0) * y100 - T(31.0);
			return T(0.10532778218603311137E0) + (T(0.41692873614065380607E-2) + (T(0.33849549774889456984E-4) + (T(0.42064596193692630143E-6) + (T(0.66494579697622432987E-8) + (T(0.13094103581931802337E-9) + (T(0.31896187409777777778E-11) + T(0.97271974184476560742E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 16:
		{
			T t = T(2.0) * y100 - T(33.0);
			return T(0.11380523107427108222E0) + (T(0.43099572287871821013E-2) + (T(0.36544324341565929930E-4) + (T(0.47965044028581857764E-6) + (T(0.81819034238463698796E-8) + (T(0.17934133239549647357E-9) + (T(0.50956666166186293627E-11) + (T(0.18850487318190638010E-12) + T(0.79697813173519853340E-14) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 17:
		{
			T t = T(2.0) * y100 - T(35.0);
			return T(0.12257529703447467345E0) + (T(0.44621675710026986366E-2) + (T(0.39634304721292440285E-4) + (T(0.55321553769873381819E-6) + (T(0.10343619428848520870E-7) + (T(0.26033830170470368088E-9) + (T(0.87743837749108025357E-11) + (T(0.34427092430230063401E-12) + T(0.10205506615709843189E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 18:
		{
			T t = T(2.0) * y100 - T(37.0);
			return T(0.13166276955656699478E0) + (T(0.46276970481783001803E-2) + (T(0.43225026380496399310E-4) + (T(0.64799164020016902656E-6) + (T(0.13580082794704641782E-7) + (T(0.39839800853954313927E-9) + (T(0.14431142411840000000E-10) + T(0.42193457308830027541E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 19:
		{
			T t = T(2.0) * y100 - T(39.0);
			return T(0.14109647869803356475E0) + (T(0.48088424418545347758E-2) + (T(0.47474504753352150205E-4) + (T(0.77509866468724360352E-6) + (T(0.18536851570794291724E-7) + (T(0.60146623257887570439E-9) + (T(0.18533978397305276318E-10) + (T(0.41033845938901048380E-13) - T(0.46160680279304825485E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 20:
		{
			T t = T(2.0) * y100 - T(41.0);
			return T(0.15091057940548936603E0) + (T(0.50086864672004685703E-2) + (T(0.52622482832192230762E-4) + (T(0.95034664722040355212E-6) + (T(0.25614261331144718769E-7) + (T(0.80183196716888606252E-9) + (T(0.12282524750534352272E-10) + (T(-0.10531774117332273617E-11) - T(0.86157181395039646412E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 21:
		{
			T t = T(2.0) * y100 - T(43.0);
			return T(0.16114648116017010770E0) + (T(0.52314661581655369795E-2) + (T(0.59005534545908331315E-4) + (T(0.11885518333915387760E-5) + (T(0.33975801443239949256E-7) + (T(0.82111547144080388610E-9) + (T(-0.12357674017312854138E-10) + (T(-0.24355112256914479176E-11) - T(0.75155506863572930844E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 22:
		{
			T t = T(2.0) * y100 - T(45.0);
			return T(0.17185551279680451144E0) + (T(0.54829002967599420860E-2) + (T(0.67013226658738082118E-4) + (T(0.14897400671425088807E-5) + (T(0.40690283917126153701E-7) + (T(0.44060872913473778318E-9) + (T(-0.52641873433280000000E-10) - T(0.30940587864543343124E-11) * t) * t) * t) * t) * t) * t) * t;
		}
		case 23:
		{
			T t = T(2.0) * y100 - T(47.0);
			return T(0.18310194559815257381E0) + (T(0.57701559375966953174E-2) + (T(0.76948789401735193483E-4) + (T(0.18227569842290822512E-5) + (T(0.41092208344387212276E-7) + (T(-0.44009499965694442143E-9) + (T(-0.92195414685628803451E-10) + (T(-0.22657389705721753299E-11) + T(0.10004784908106839254E-12) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 24:
		{
			T t = T(2.0) * y100 - T(49.0);
			return T(0.19496527191546630345E0) + (T(0.61010853144364724856E-2) + (T(0.88812881056342004864E-4) + (T(0.21180686746360261031E-5) + (T(0.30652145555130049203E-7) + (T(-0.16841328574105890409E-8) + (T(-0.11008129460612823934E-9) + (T(-0.12180794204544515779E-12) + T(0.15703325634590334097E-12) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 25:
		{
			T t = T(2.0) * y100 - T(51.0);
			return T(0.20754006813966575720E0) + (T(0.64825787724922073908E-2) + (T(0.10209599627522311893E-3) + (T(0.22785233392557600468E-5) + (T(0.73495224449907568402E-8) + (T(-0.29442705974150112783E-8) + (T(-0.94082603434315016546E-10) + (T(0.23609990400179321267E-11) + T(0.14141908654269023788E-12) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 26:
		{
			T t = T(2.0) * y100 - T(53.0);
			return T(0.22093185554845172146E0) + (T(0.69182878150187964499E-2) + (T(0.11568723331156335712E-3) + (T(0.22060577946323627739E-5) + (T(-0.26929730679360840096E-7) + (T(-0.38176506152362058013E-8) + (T(-0.47399503861054459243E-10) + (T(0.40953700187172127264E-11) + T(0.69157730376118511127E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 27:
		{
			T t = T(2.0) * y100 - T(55.0);
			return T(0.23524827304057813918E0) + (T(0.74063350762008734520E-2) + (T(0.12796333874615790348E-3) + (T(0.18327267316171054273E-5) + (T(-0.66742910737957100098E-7) + (T(-0.40204740975496797870E-8) + (T(0.14515984139495745330E-10) + (T(0.44921608954536047975E-11) - T(0.18583341338983776219E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 28:
		{
			T t = T(2.0) * y100 - T(57.0);
			return T(0.25058626331812744775E0) + (T(0.79377285151602061328E-2) + (T(0.13704268650417478346E-3) + (T(0.11427511739544695861E-5) + (T(-0.10485442447768377485E-6) + (T(-0.34850364756499369763E-8) + (T(0.72656453829502179208E-10) + (T(0.36195460197779299406E-11) - T(0.84882136022200714710E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 29:
		{
			T t = T(2.0) * y100 - T(59.0);
			return T(0.26701724900280689785E0) + (T(0.84959936119625864274E-2) + (T(0.14112359443938883232E-3) + (T(0.17800427288596909634E-6) + (T(-0.13443492107643109071E-6) + (T(-0.23512456315677680293E-8) + (T(0.11245846264695936769E-9) + (T(0.19850501334649565404E-11) - T(0.11284666134635050832E-12) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 30:
		{
			T t = T(2.0) * y100 - T(61.0);
			return T(0.28457293586253654144E0) + (T(0.90581563892650431899E-2) + (T(0.13880520331140646738E-3) + (T(-0.97262302362522896157E-6) + (T(-0.15077100040254187366E-6) + (T(-0.88574317464577116689E-9) + (T(0.12760311125637474581E-9) + (T(0.20155151018282695055E-12) - T(0.10514169375181734921E-12) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 31:
		{
			T t = T(2.0) * y100 - T(63.0);
			return T(0.30323425595617385705E0) + (T(0.95968346790597422934E-2) + (T(0.12931067776725883939E-3) + (T(-0.21938741702795543986E-5) + (T(-0.15202888584907373963E-6) + (T(0.61788350541116331411E-9) + (T(0.11957835742791248256E-9) + (T(-0.12598179834007710908E-11) - T(0.75151817129574614194E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 32:
		{
			T t = T(2.0) * y100 - T(65.0);
			return T(0.32292521181517384379E0) + (T(0.10082957727001199408E-1) + (T(0.11257589426154962226E-3) + (T(-0.33670890319327881129E-5) + (T(-0.13910529040004008158E-6) + (T(0.19170714373047512945E-8) + (T(0.94840222377720494290E-10) + (T(-0.21650018351795353201E-11) - T(0.37875211678024922689E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 33:
		{
			T t = T(2.0) * y100 - T(67.0);
			return T(0.34351233557911753862E0) + (T(0.10488575435572745309E-1) + (T(0.89209444197248726614E-4) + (T(-0.43893459576483345364E-5) + (T(-0.11488595830450424419E-6) + (T(0.28599494117122464806E-8) + (T(0.61537542799857777779E-10) - T(0.24935749227658002212E-11) * t) * t) * t) * t) * t) * t) * t;
		}
		case 34:
		{
			T t = T(2.0) * y100 - T(69.0);
			return T(0.36480946642143669093E0) + (T(0.10789304203431861366E-1) + (T(0.60357993745283076834E-4) + (T(-0.51855862174130669389E-5) + (T(-0.83291664087289801313E-7) + (T(0.33898011178582671546E-8) + (T(0.27082948188277716482E-10) + (T(-0.23603379397408694974E-11) + T(0.19328087692252869842E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 35:
		{
			T t = T(2.0) * y100 - T(71.0);
			return T(0.38658679935694939199E0) + (T(0.10966119158288804999E-1) + (T(0.27521612041849561426E-4) + (T(-0.57132774537670953638E-5) + (T(-0.48404772799207914899E-7) + (T(0.35268354132474570493E-8) + (T(-0.32383477652514618094E-11) + (T(-0.19334202915190442501E-11) + T(0.32333189861286460270E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 36:
		{
			T t = T(2.0) * y100 - T(73.0);
			return T(0.40858275583808707870E0) + (T(0.11006378016848466550E-1) + (T(-0.76396376685213286033E-5) + (T(-0.59609835484245791439E-5) + (T(-0.13834610033859313213E-7) + (T(0.33406952974861448790E-8) + (T(-0.26474915974296612559E-10) + (T(-0.13750229270354351983E-11) + T(0.36169366979417390637E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 37:
		{
			T t = T(2.0) * y100 - T(75.0);
			return T(0.43051714914006682977E0) + (T(0.10904106549500816155E-1) + (T(-0.43477527256787216909E-4) + (T(-0.59429739547798343948E-5) + (T(0.17639200194091885949E-7) + (T(0.29235991689639918688E-8) + (T(-0.41718791216277812879E-10) + (T(-0.81023337739508049606E-12) + T(0.33618915934461994428E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 38:
		{
			T t = T(2.0) * y100 - T(77.0);
			return T(0.45210428135559607406E0) + (T(0.10659670756384400554E-1) + (T(-0.78488639913256978087E-4) + (T(-0.56919860886214735936E-5) + (T(0.44181850467477733407E-7) + (T(0.23694306174312688151E-8) + (T(-0.49492621596685443247E-10) + (T(-0.31827275712126287222E-12) + T(0.27494438742721623654E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 39:
		{
			T t = T(2.0) * y100 - T(79.0);
			return T(0.47306491195005224077E0) + (T(0.10279006119745977570E-1) + (T(-0.11140268171830478306E-3) + (T(-0.52518035247451432069E-5) + (T(0.64846898158889479518E-7) + (T(0.17603624837787337662E-8) + (T(-0.51129481592926104316E-10) + (T(0.62674584974141049511E-13) + T(0.20055478560829935356E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 40:
		{
			T t = T(2.0) * y100 - T(81.0);
			return T(0.49313638965719857647E0) + (T(0.97725799114772017662E-2) + (T(-0.14122854267291533334E-3) + (T(-0.46707252568834951907E-5) + (T(0.79421347979319449524E-7) + (T(0.11603027184324708643E-8) + (T(-0.48269605844397175946E-10) + (T(0.32477251431748571219E-12) + T(0.12831052634143527985E-13) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 41:
		{
			T t = T(2.0) * y100 - T(83.0);
			return T(0.51208057433416004042E0) + (T(0.91542422354009224951E-2) + (T(-0.16726530230228647275E-3) + (T(-0.39964621752527649409E-5) + (T(0.88232252903213171454E-7) + (T(0.61343113364949928501E-9) + (T(-0.42516755603130443051E-10) + (T(0.47910437172240209262E-12) + T(0.66784341874437478953E-14) * t) * t) * t) * t) * t) * t) * t) * t;
		}
		case 42:
		{
			T t = T(2.0) * y100 - T(85.0);
			return T(0.52968945458607484524E0) + (T(0.84400880445116786088E-2) + (T(-0.18908729783854258774E-3) + (T(-0.32725905467782951931E-5) + (T(0.91956190588652090659E-7) + (T(0.14593989152420122909E-9) + (T(-0.35239490687644444445E-10) + T(0.54613829888448694898E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 43:
		{
			T t = T(2.0) * y100 - T(87.0);
			return T(0.54578857454330070965E0) + (T(0.76474155195880295311E-2) + (T(-0.20651230590808213884E-3) + (T(-0.25364339140543131706E-5) + (T(0.91455367999510681979E-7) + (T(-0.23061359005297528898E-9) + (T(-0.27512928625244444444E-10) + T(0.54895806008493285579E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 44:
		{
			T t = T(2.0) * y100 - T(89.0);
			return T(0.56023851910298493910E0) + (T(0.67938321739997196804E-2) + (T(-0.21956066613331411760E-3) + (T(-0.18181127670443266395E-5) + (T(0.87650335075416845987E-7) + (T(-0.51548062050366615977E-9) + (T(-0.20068462174044444444E-10) + T(0.50912654909758187264E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 45:
		{
			T t = T(2.0) * y100 - T(91.0);
			return T(0.57293478057455721150E0) + (T(0.58965321010394044087E-2) + (T(-0.22841145229276575597E-3) + (T(-0.11404605562013443659E-5) + (T(0.81430290992322326296E-7) + (T(-0.71512447242755357629E-9) + (T(-0.13372664928000000000E-10) + T(0.44461498336689298148E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 46:
		{
			T t = T(2.0) * y100 - T(93.0);
			return T(0.58380635448407827360E0) + (T(0.49717469530842831182E-2) + (T(-0.23336001540009645365E-3) + (T(-0.51952064448608850822E-6) + (T(0.73596577815411080511E-7) + (T(-0.84020916763091566035E-9) + (T(-0.76700972702222222221E-11) + T(0.36914462807972467044E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 47:
		{
			T t = T(2.0) * y100 - T(95.0);
			return T(0.59281340237769489597E0) + (T(0.40343592069379730568E-2) + (T(-0.23477963738658326185E-3) + (T(0.34615944987790224234E-7) + (T(0.64832803248395814574E-7) + (T(-0.90329163587627007971E-9) + (T(-0.30421940400000000000E-11) + T(0.29237386653743536669E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 48:
		{
			T t = T(2.0) * y100 - T(97.0);
			return T(0.59994428743114271918E0) + (T(0.30976579788271744329E-2) + (T(-0.23308875765700082835E-3) + (T(0.51681681023846925160E-6) + (T(0.55694594264948268169E-7) + (T(-0.91719117313243464652E-9) + (T(0.53982743680000000000E-12) + T(0.22050829296187771142E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 49:
		{
			T t = T(2.0) * y100 - T(99.0);
			return T(0.60521224471819875444E0) + (T(0.21732138012345456060E-2) + (T(-0.22872428969625997456E-3) + (T(0.92588959922653404233E-6) + (T(0.46612665806531930684E-7) + (T(-0.89393722514414153351E-9) + (T(0.31718550353777777778E-11) + T(0.15705458816080549117E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 50:
		{
			T t = T(2.0) * y100 - T(101.0);
			return T(0.60865189969791123620E0) + (T(0.12708480848877451719E-2) + (T(-0.22212090111534847166E-3) + (T(0.12636236031532793467E-5) + (T(0.37904037100232937574E-7) + (T(-0.84417089968101223519E-9) + (T(0.49843180828444444445E-11) + T(0.10355439441049048273E-12) * t) * t) * t) * t) * t) * t) * t;
		}
		case 51:
		{
			T t = T(2.0) * y100 - T(103.0);
			return T(0.61031580103499200191E0) + (T(0.39867436055861038223E-3) + (T(-0.21369573439579869291E-3) + (T(0.15339402129026183670E-5) + (T(0.29787479206646594442E-7) + (T(-0.77687792914228632974E-9) + (T(0.61192452741333333334E-11) + T(0.60216691829459295780E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 52:
		{
			T t = T(2.0) * y100 - T(105.0);
			return T(0.61027109047879835868E0) + (T(-0.43680904508059878254E-3) + (T(-0.20383783788303894442E-3) + (T(0.17421743090883439959E-5) + (T(0.22400425572175715576E-7) + (T(-0.69934719320045128997E-9) + (T(0.67152759655111111110E-11) + T(0.26419960042578359995E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 53:
		{
			T t = T(2.0) * y100 - T(107.0);
			return T(0.60859639489217430521E0) + (T(-0.12305921390962936873E-2) + (T(-0.19290150253894682629E-3) + (T(0.18944904654478310128E-5) + (T(0.15815530398618149110E-7) + (T(-0.61726850580964876070E-9) + T(0.68987888999111111110E-11) * t) * t) * t) * t) * t) * t;
		}
		case 54:
		{
			T t = T(2.0) * y100 - T(109.0);
			return T(0.60537899426486075181E0) + (T(-0.19790062241395705751E-2) + (T(-0.18120271393047062253E-3) + (T(0.19974264162313241405E-5) + (T(0.10055795094298172492E-7) + (T(-0.53491997919318263593E-9) + (T(0.67794550295111111110E-11) - T(0.17059208095741511603E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 55:
		{
			T t = T(2.0) * y100 - T(111.0);
			return T(0.60071229457904110537E0) + (T(-0.26795676776166354354E-2) + (T(-0.16901799553627508781E-3) + (T(0.20575498324332621581E-5) + (T(0.51077165074461745053E-8) + (T(-0.45536079828057221858E-9) + (T(0.64488005516444444445E-11) - T(0.29311677573152766338E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 56:
		{
			T t = T(2.0) * y100 - T(113.0);
			return T(0.59469361520112714738E0) + (T(-0.33308208190600993470E-2) + (T(-0.15658501295912405679E-3) + (T(0.20812116912895417272E-5) + (T(0.93227468760614182021E-9) + (T(-0.38066673740116080415E-9) + (T(0.59806790359111111110E-11) - T(0.36887077278950440597E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 57:
		{
			T t = T(2.0) * y100 - T(115.0);
			return T(0.58742228631775388268E0) + (T(-0.39321858196059227251E-2) + (T(-0.14410441141450122535E-3) + (T(0.20743790018404020716E-5) + (T(-0.25261903811221913762E-8) + (T(-0.31212416519526924318E-9) + (T(0.54328422462222222221E-11) - T(0.40864152484979815972E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 58:
		{
			T t = T(2.0) * y100 - T(117.0);
			return T(0.57899804200033018447E0) + (T(-0.44838157005618913447E-2) + (T(-0.13174245966501437965E-3) + (T(0.20425306888294362674E-5) + (T(-0.53330296023875447782E-8) + (T(-0.25041289435539821014E-9) + (T(0.48490437205333333334E-11) - T(0.42162206939169045177E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 59:
		{
			T t = T(2.0) * y100 - T(119.0);
			return T(0.56951968796931245974E0) + (T(-0.49864649488074868952E-2) + (T(-0.11963416583477567125E-3) + (T(0.19906021780991036425E-5) + (T(-0.75580140299436494248E-8) + (T(-0.19576060961919820491E-9) + (T(0.42613011928888888890E-11) - T(0.41539443304115604377E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 60:
		{
			T t = T(2.0) * y100 - T(121.0);
			return T(0.55908401930063918964E0) + (T(-0.54413711036826877753E-2) + (T(-0.10788661102511914628E-3) + (T(0.19229663322982839331E-5) + (T(-0.92714731195118129616E-8) + (T(-0.14807038677197394186E-9) + (T(0.36920870298666666666E-11) - T(0.39603726688419162617E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 61:
		{
			T t = T(2.0) * y100 - T(123.0);
			return T(0.54778496152925675315E0) + (T(-0.58501497933213396670E-2) + (T(-0.96582314317855227421E-4) + (T(0.18434405235069270228E-5) + (T(-0.10541580254317078711E-7) + (T(-0.10702303407788943498E-9) + (T(0.31563175582222222222E-11) - T(0.36829748079110481422E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 62:
		{
			T t = T(2.0) * y100 - T(125.0);
			return T(0.53571290831682823999E0) + (T(-0.62147030670760791791E-2) + (T(-0.85782497917111760790E-4) + (T(0.17553116363443470478E-5) + (T(-0.11432547349815541084E-7) + (T(-0.72157091369041330520E-10) + (T(0.26630811607111111111E-11) - T(0.33578660425893164084E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 63:
		{
			T t = T(2.0) * y100 - T(127.0);
			return T(0.52295422962048434978E0) + (T(-0.65371404367776320720E-2) + (T(-0.75530164941473343780E-4) + (T(0.16613725797181276790E-5) + (T(-0.12003521296598910761E-7) + (T(-0.42929753689181106171E-10) + (T(0.22170894940444444444E-11) - T(0.30117697501065110505E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 64:
		{
			T t = T(2.0) * y100 - T(129.0);
			return T(0.50959092577577886140E0) + (T(-0.68197117603118591766E-2) + (T(-0.65852936198953623307E-4) + (T(0.15639654113906716939E-5) + (T(-0.12308007991056524902E-7) + (T(-0.18761997536910939570E-10) + (T(0.18198628922666666667E-11) - T(0.26638355362285200932E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 65:
		{
			T t = T(2.0) * y100 - T(131.0);
			return T(0.49570040481823167970E0) + (T(-0.70647509397614398066E-2) + (T(-0.56765617728962588218E-4) + (T(0.14650274449141448497E-5) + (T(-0.12393681471984051132E-7) + (T(0.92904351801168955424E-12) + (T(0.14706755960177777778E-11) - T(0.23272455351266325318E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 66:
		{
			T t = T(2.0) * y100 - T(133.0);
			return T(0.48135536250935238066E0) + (T(-0.72746293327402359783E-2) + (T(-0.48272489495730030780E-4) + (T(0.13661377309113939689E-5) + (T(-0.12302464447599382189E-7) + (T(0.16707760028737074907E-10) + (T(0.11672928324444444444E-11) - T(0.20105801424709924499E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 67:
		{
			T t = T(2.0) * y100 - T(135.0);
			return T(0.46662374675511439448E0) + (T(-0.74517177649528487002E-2) + (T(-0.40369318744279128718E-4) + (T(0.12685621118898535407E-5) + (T(-0.12070791463315156250E-7) + (T(0.29105507892605823871E-10) + (T(0.90653314645333333334E-12) - T(0.17189503312102982646E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 68:
		{
			T t = T(2.0) * y100 - T(137.0);
			return T(0.45156879030168268778E0) + (T(-0.75983560650033817497E-2) + (T(-0.33045110380705139759E-4) + (T(0.11732956732035040896E-5) + (T(-0.11729986947158201869E-7) + (T(0.38611905704166441308E-10) + (T(0.68468768305777777779E-12) - T(0.14549134330396754575E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 69:
		{
			T t = T(2.0) * y100 - T(139.0);
			return T(0.43624909769330896904E0) + (T(-0.77168291040309554679E-2) + (T(-0.26283612321339907756E-4) + (T(0.10811018836893550820E-5) + (T(-0.11306707563739851552E-7) + (T(0.45670446788529607380E-10) + (T(0.49782492549333333334E-12) - T(0.12191983967561779442E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 70:
		{
			T t = T(2.0) * y100 - T(141.0);
			return T(0.42071877443548481181E0) + (T(-0.78093484015052730097E-2) + (T(-0.20064596897224934705E-4) + (T(0.99254806680671890766E-6) + (T(-0.10823412088884741451E-7) + (T(0.50677203326904716247E-10) + (T(0.34200547594666666666E-12) - T(0.10112698698356194618E-13) * t) * t) * t) * t) * t) * t) * t;
		}
		case 71:
		{
			T t = T(2.0) * y100 - T(143.0);
			return T(0.40502758809710844280E0) + (T(-0.78780384460872937555E-2) + (T(-0.14364940764532853112E-4) + (T(0.90803709228265217384E-6) + (T(-0.10298832847014466907E-7) + (T(0.53981671221969478551E-10) + (T(0.21342751381333333333E-12) - T(0.82975901848387729274E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 72:
		{
			T t = T(2.0) * y100 - T(145.0);
			return T(0.38922115269731446690E0) + (T(-0.79249269708242064120E-2) + (T(-0.91595258799106970453E-5) + (T(0.82783535102217576495E-6) + (T(-0.97484311059617744437E-8) + (T(0.55889029041660225629E-10) + (T(0.10851981336888888889E-12) - T(0.67278553237853459757E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 73:
		{
			T t = T(2.0) * y100 - T(147.0);
			return T(0.37334112915460307335E0) + (T(-0.79519385109223148791E-2) + (T(-0.44219833548840469752E-5) + (T(0.75209719038240314732E-6) + (T(-0.91848251458553190451E-8) + (T(0.56663266668051433844E-10) + (T(0.23995894257777777778E-13) - T(0.53819475285389344313E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 74:
		{
			T t = T(2.0) * y100 - T(149.0);
			return T(0.35742543583374223085E0) + (T(-0.79608906571527956177E-2) + (T(-0.12530071050975781198E-6) + (T(0.68088605744900552505E-6) + (T(-0.86181844090844164075E-8) + (T(0.56530784203816176153E-10) + (T(-0.43120012248888888890E-13) - T(0.42372603392496813810E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 75:
		{
			T t = T(2.0) * y100 - T(151.0);
			return T(0.34150846431979618536E0) + (T(-0.79534924968773806029E-2) + (T(0.37576885610891515813E-5) + (T(0.61419263633090524326E-6) + (T(-0.80565865409945960125E-8) + (T(0.55684175248749269411E-10) + (T(-0.95486860764444444445E-13) - T(0.32712946432984510595E-14) * t) * t) * t) * t) * t) * t) * t;
		}
		case 76:
		{
			T t = T(2.0) * y100 - T(153.0);
			return T(0.32562129649136346824E0) + (T(-0.79313448067948884309E-2) + (T(0.72539159933545300034E-5) + (T(0.55195028297415503083E-6) + (T(-0.75063365335570475258E-8) + (T(0.54281686749699595941E-10) - T(0.13545424295111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 77:
		{
			T t = T(2.0) * y100 - T(155.0);
			return T(0.30979191977078391864E0) + (T(-0.78959416264207333695E-2) + (T(0.10389774377677210794E-4) + (T(0.49404804463196316464E-6) + (T(-0.69722488229411164685E-8) + (T(0.52469254655951393842E-10) - T(0.16507860650666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 78:
		{
			T t = T(2.0) * y100 - T(157.0);
			return T(0.29404543811214459904E0) + (T(-0.78486728990364155356E-2) + (T(0.13190885683106990459E-4) + (T(0.44034158861387909694E-6) + (T(-0.64578942561562616481E-8) + (T(0.50354306498006928984E-10) - T(0.18614473550222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 79:
		{
			T t = T(2.0) * y100 - T(159.0);
			return T(0.27840427686253660515E0) + (T(-0.77908279176252742013E-2) + (T(0.15681928798708548349E-4) + (T(0.39066226205099807573E-6) + (T(-0.59658144820660420814E-8) + (T(0.48030086420373141763E-10) - T(0.20018995173333333333E-12) * t) * t) * t) * t) * t) * t;
		}
		case 80:
		{
			T t = T(2.0) * y100 - T(161.0);
			return T(0.26288838011163800908E0) + (T(-0.77235993576119469018E-2) + (T(0.17886516796198660969E-4) + (T(0.34482457073472497720E-6) + (T(-0.54977066551955420066E-8) + (T(0.45572749379147269213E-10) - T(0.20852924954666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 81:
		{
			T t = T(2.0) * y100 - T(163.0);
			return T(0.24751539954181029717E0) + (T(-0.76480877165290370975E-2) + (T(0.19827114835033977049E-4) + (T(0.30263228619976332110E-6) + (T(-0.50545814570120129947E-8) + (T(0.43043879374212005966E-10) - T(0.21228012028444444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 82:
		{
			T t = T(2.0) * y100 - T(165.0);
			return T(0.23230087411688914593E0) + (T(-0.75653060136384041587E-2) + (T(0.21524991113020016415E-4) + (T(0.26388338542539382413E-6) + (T(-0.46368974069671446622E-8) + (T(0.40492715758206515307E-10) - T(0.21238627815111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 83:
		{
			T t = T(2.0) * y100 - T(167.0);
			return T(0.21725840021297341931E0) + (T(-0.74761846305979730439E-2) + (T(0.23000194404129495243E-4) + (T(0.22837400135642906796E-6) + (T(-0.42446743058417541277E-8) + (T(0.37958104071765923728E-10) - T(0.20963978568888888889E-12) * t) * t) * t) * t) * t) * t;
		}
		case 84:
		{
			T t = T(2.0) * y100 - T(169.0);
			return T(0.20239979200788191491E0) + (T(-0.73815761980493466516E-2) + (T(0.24271552727631854013E-4) + (T(0.19590154043390012843E-6) + (T(-0.38775884642456551753E-8) + (T(0.35470192372162901168E-10) - T(0.20470131678222222222E-12) * t) * t) * t) * t) * t) * t;
		}
		case 85:
		{
			T t = T(2.0) * y100 - T(171.0);
			return T(0.18773523211558098962E0) + (T(-0.72822604530339834448E-2) + (T(0.25356688567841293697E-4) + (T(0.16626710297744290016E-6) + (T(-0.35350521468015310830E-8) + (T(0.33051896213898864306E-10) - T(0.19811844544000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 86:
		{
			T t = T(2.0) * y100 - T(173.0);
			return T(0.17327341258479649442E0) + (T(-0.71789490089142761950E-2) + (T(0.26272046822383820476E-4) + (T(0.13927732375657362345E-6) + (T(-0.32162794266956859603E-8) + (T(0.30720156036105652035E-10) - T(0.19034196304000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 87:
		{
			T t = T(2.0) * y100 - T(175.0);
			return T(0.15902166648328672043E0) + (T(-0.70722899934245504034E-2) + (T(0.27032932310132226025E-4) + (T(0.11474573347816568279E-6) + (T(-0.29203404091754665063E-8) + (T(0.28487010262547971859E-10) - T(0.18174029063111111111E-12) * t) * t) * t) * t) * t) * t;
		}
		case 88:
		{
			T t = T(2.0) * y100 - T(177.0);
			return T(0.14498609036610283865E0) + (T(-0.69628725220045029273E-2) + (T(0.27653554229160596221E-4) + (T(0.92493727167393036470E-7) + (T(-0.26462055548683583849E-8) + (T(0.26360506250989943739E-10) - T(0.17261211260444444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 89:
		{
			T t = T(2.0) * y100 - T(179.0);
			return T(0.13117165798208050667E0) + (T(-0.68512309830281084723E-2) + (T(0.28147075431133863774E-4) + (T(0.72351212437979583441E-7) + (T(-0.23927816200314358570E-8) + (T(0.24345469651209833155E-10) - T(0.16319736960000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 90:
		{
			T t = T(2.0) * y100 - T(181.0);
			return T(0.11758232561160626306E0) + (T(-0.67378491192463392927E-2) + (T(0.28525664781722907847E-4) + (T(0.54156999310046790024E-7) + (T(-0.21589405340123827823E-8) + (T(0.22444150951727334619E-10) - T(0.15368675584000000000E-12) * t) * t) * t) * t) * t) * t;
		}
		case 91:
		{
			T t = T(2.0) * y100 - T(183.0);
			return T(0.10422112945361673560E0) + (T(-0.66231638959845581564E-2) + (T(0.28800551216363918088E-4) + (T(0.37758983397952149613E-7) + (T(-0.19435423557038933431E-8) + (T(0.20656766125421362458E-10) - T(0.14422990012444444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 92:
		{
			T t = T(2.0) * y100 - T(185.0);
			return T(0.91090275493541084785E-1) + (T(-0.65075691516115160062E-2) + (T(0.28982078385527224867E-4) + (T(0.23014165807643012781E-7) + (T(-0.17454532910249875958E-8) + (T(0.18981946442680092373E-10) - T(0.13494234691555555556E-12) * t) * t) * t) * t) * t) * t;
		}
		case 93:
		{
			T t = T(2.0) * y100 - T(187.0);
			return T(0.78191222288771379358E-1) + (T(-0.63914190297303976434E-2) + (T(0.29079759021299682675E-4) + (T(0.97885458059415717014E-8) + (T(-0.15635596116134296819E-8) + (T(0.17417110744051331974E-10) - T(0.12591151763555555556E-12) * t) * t) * t) * t) * t) * t;
		}
		case 94:
		{
			T t = T(2.0) * y100 - T(189.0);
			return T(0.65524757106147402224E-1) + (T(-0.62750311956082444159E-2) + (T(0.29102328354323449795E-4) + (T(-0.20430838882727954582E-8) + (T(-0.13967781903855367270E-8) + (T(0.15958771833747057569E-10) - T(0.11720175765333333333E-12) * t) * t) * t) * t) * t) * t;
		}
		case 95:
		{
			T t = T(2.0) * y100 - T(191.0);
			return T(0.53091065838453612773E-1) + (T(-0.61586898417077043662E-2) + (T(0.29057796072960100710E-4) + (T(-0.12597414620517987536E-7) + (T(-0.12440642607426861943E-8) + (T(0.14602787128447932137E-10) - T(0.10885859114666666667E-12) * t) * t) * t) * t) * t) * t;
		}
		case 96:
		{
			T t = T(2.0) * y100 - T(193.0);
			return T(0.40889797115352738582E-1) + (T(-0.60426484889413678200E-2) + (T(0.28953496450191694606E-4) + (T(-0.21982952021823718400E-7) + (T(-0.11044169117553026211E-8) + (T(0.13344562332430552171E-10) - T(0.10091231402844444444E-12) * t) * t) * t) * t) * t) * t;
		}
		case 97: case 98: case 99: case 100:
		{ // use Taylor expansion for small x (|x| <= 0.0309...) (2/sqrt(pi)) * (x - 2/3 x^3  + 4/15 x^5  - 8/105 x^7 + 16/945 x^9) 
			T x2 = x * x;
			return x * (T(1.1283791670955125739) - x2 * (T(0.75225277806367504925) - x2 * (T(0.30090111122547001970) - x2 * (T(0.085971746064420005629) - x2 * T(0.016931216931216931217)))));
		}
	}
	
	return T(std::numeric_limits<double>::quiet_NaN());
}

template <typename T> inline T w_im(T x)
{
	double val_x = get_value(x);
	const double ispi = 0.56418958354775628694807945156;

	if (val_x >= 0.0)
	{
		if (val_x > 45.0)
		{
			if (val_x > 5.0E7)
			{
				return T(ispi) / x;
			}
			
			T x2 = x * x;
			return T(ispi) * (x2 * (x2 - T(4.5)) + T(2.0)) / (x * (x2 * (x2 - T(5.0)) + T(3.75)));
		}
		
		return w_im_y100(T(100.0) / (T(1.0) + x), x);
	}
	else
	{
		if (val_x < -45.0)
		{
			if (val_x < -5.0E7)
			{
				return T(ispi) / x;
			}
			
			T x2 = x * x;
			return T(ispi) * (x2 * (x2 - T(4.5)) + T(2.0)) / (x * (x2 * (x2 - T(5.0)) + T(3.75)));
		}

		return -w_im_y100(T(100.0) / (T(1.0) - x), -x);
	}
}

static const double expa2n2[] = {
	7.64405281671221563E-01, 3.41424527166548425E-01, 8.91072646929412548E-02,
	1.35887299055460086E-02, 1.21085455253437481E-03, 6.30452613933449404E-05,
	1.91805156577114683E-06, 3.40969447714832381E-08, 3.54175089099469393E-10,
	2.14965079583260682E-12, 7.62368911833724354E-15, 1.57982797110681093E-17,
	1.91294189103582677E-20, 1.35344656764205340E-23, 5.59535712428588720E-27,
	1.35164257972401769E-30, 1.90784582843501167E-34, 1.57351920291442930E-38,
	7.58312432328032845E-43, 2.13536275438697082E-47, 3.51352063787195769E-52,
	3.37800830266396920E-57, 1.89769439468301000E-62, 6.22929926072668851E-68,
	1.19481172006938722E-73, 1.33908181133005953E-79, 8.76924303483223939E-86,
	3.35555576166254986E-92, 7.50264110688173024E-99, 9.80192200745410268E-106,
	7.48265412822268959E-113, 3.33770122566809425E-120, 8.69934598159861140E-128,
	1.32486951484088852E-135, 1.17898144201315253E-143, 6.13039120236180012E-152,
	1.86258785950822098E-160, 3.30668408201432783E-169, 3.43017280887946235E-178,
	2.07915397775808219E-187, 7.36384545323984966E-197, 1.52394760394085741E-206,
	1.84281935046532100E-216, 1.30209553802992923E-226, 5.37588903521080531E-237,
	1.29689584599763145E-247, 1.82813078022866562E-258, 1.50576355348684241E-269,
	7.24692320799294194E-281, 2.03797051314726829E-292, 3.34880215927873807E-304,
	0.0
};

template <typename T> inline T erfcx(T x)
{
	double val_x = get_value(x);
	const double ispi = 0.56418958354775628694807945156;

	if (val_x >= 0.0)
	{
		if (val_x > 50.0)
		{
			if (val_x > 5.0E7)
			{
				return T(ispi) / x;
			}

			T x2 = x * x;
			return T(ispi) * (x2 * (x2 + T(4.5)) + T(2.0)) / (x * (x2 * (x2 + T(5.0)) + T(3.75)));
		}

		return erfcx_y100(T(400.0) / (T(4.0) + x));
	}
	else 
	{
		if (val_x < -26.7) 
		{
			return T(std::numeric_limits<double>::infinity());
		}

		T x2 = x * x;
		using std::exp;
		
		if (val_x < -6.1) 
		{
			return T(2.0) * exp(x2);
		} 
		else 
		{
			return T(2.0) * exp(x2) - erfcx_y100(T(400.0) / (T(4.0) - x));
		}
	}
}

template <typename T> inline complex<T> faddeeva(complex<T> z, T relerr = T(0.0))
{
	using std::abs;
	using std::exp;
	using std::sqrt;
	using std::log;
	using std::sin;
	using std::cos;
	using std::floor;
	using std::copysign;
	using std::isnan;
	using std::isinf;

	const double eps = std::numeric_limits<double>::epsilon();
	const double pi = std::numbers::pi;
	const T ispi = T(0.56418958354775628694807945156); // 1/sqrt(pi)

	// 純虚数または実数の場合の特殊処理
	if (get_value(z.re) == 0.0)
	{
		return complex<T>(erfcx(z.im), z.re);
	}
	else if (get_value(z.im) == 0.0)
	{
		return complex<T>(exp(-sqr(z.re)), w_im(z.re));
	}

	T a, a2, c;
	double v_relerr = get_value(relerr);

	if (v_relerr <= eps)
	{
		v_relerr = eps;
		a = T(0.518321480430085929872);
		c = T(0.329973702884629072537);
		a2 = T(0.268657157075235951582);
	}
	else
	{
		if (v_relerr > 0.1)
		{
			v_relerr = 0.1;
		}

		a = T(pi) / sqrt(-log(T(v_relerr * 0.5)));
		c = (T(2.0) / T(pi)) * T(a);
		a2 = T(a) * T(a);
	}

	const T x = abs(z.re);
	const T y = z.im;
	const T ya = abs(y);

	const double vx = get_value(x);
	const double vy = get_value(y);
	const double vya = get_value(ya);

	complex<T> ret = T(0.0);
	T sum1 = T(0.0);
	T sum2 = T(0.0);
	T sum3 = T(0.0);
	T sum4 = T(0.0);
	T sum5 = T(0.0);

	// アルゴリズムの領域選択
	if (vya > 7.0 || (vx > 6.0 && (vya > 0.1 || (vx > 8.0 && vya > 1.0E-10) || vx > 28.0)))
	{
		T xs = (vy < 0.0) ? -z.re : z.re;

		if (get_value(x + ya) > 4000.0)
		{
			if (get_value(x + ya) > 1.0E7)
			{
				if (vx > vya)
				{
					T yax = ya / xs;
					T denom = T(ispi) / (xs + yax * ya);
					ret = complex<T>(denom * yax, denom);
				}
				else if (isinf(ya))
				{
					return ((isnan(x) || vy < 0.0) ? complex<T>(T(std::numeric_limits<double>::quiet_NaN()), T(std::numeric_limits<double>::quiet_NaN())) : complex<T>(T(0.0), T(0.0)));
				}
				else
				{
					T xya = xs / ya;
					T denom = T(ispi) / (xya * xs + ya);
					ret = complex<T>(denom, denom * xya);
				}
			}
			else
			{
				T dr = xs * xs - ya * ya - T(0.5);
				T di = T(2.0) * xs * ya;
				T denom = T(ispi) / (dr * dr + di * di);
				ret = complex<T>(denom * (xs * di - ya * dr), denom * (xs * dr + ya * di));
			}
		}
		else
		{
			T nu = floor(T(3.9) + T(11.398) / (T(0.08254) * x + T(0.1421) * ya + T(0.2023)));
			T wr = xs;
			T wi = ya;

			for (double n_val = 0.5 * (get_value(nu) - 1.0); n_val > 0.4; n_val -= 0.5)
			{
				T denom = n_val / (wr * wr + wi * wi);
				wr = xs - wr * denom;
				wi = ya + wi * denom;
			}

			T denom = T(ispi) / (wr * wr + wi * wi);
			ret = complex<T>(denom * wi, denom * wr);
		}

		if (vy < 0.0)
		{
			return T(2.0) * exp(complex<T>((ya - x) * (x + ya), T(2.0) * xs * y)) - ret;
		}
		else
		{
			return ret;
		}
	}
	else if (vx < 10.0) 
	{
		T prod2ax = T(1.0);
		T prodm2ax = T(1.0);
		T expx2;

		if (isnan(ya)) return complex<T>(ya, ya);

		if (get_value(relerr) <= eps)
		{
			if (vx < 5.0E-4)
			{
				T x2 = x * x;
				expx2 = T(1.0) - x2 * (T(1.0) - T(0.5) * x2);
				T ax2 = T(1.036642960860171859744) * x;
				T exp2ax = T(1.0) + ax2 * (T(1.0) + ax2 * (T(0.5) + T(0.166666666666666666667) * ax2));
				T expm2ax = T(1.0) - ax2 * (T(1.0) - ax2 * (T(0.5) - T(0.166666666666666666667) * ax2));

				for (int n = 1; ; n++)
				{
					T coef = T(expa2n2[n - 1]) * expx2 / (a2 * (double(n) * n) + y * y);
					prod2ax *= exp2ax;
					prodm2ax *= expm2ax;

					sum1 += coef;
					sum2 += coef * prodm2ax;
					sum3 += coef * prod2ax;
					sum5 += coef * (T(2.0) * a) * double(n) * sinh_taylor(T(2.0) * a * T(double(n)) * x);
					
					if (get_value(coef * prod2ax) < v_relerr * get_value(sum3)) break;
				}
			}
			else
			{
				expx2 = exp(-x * x);
				T exp2ax = exp((T(2.0) * a) * x);
				T expm2ax = T(1.0) / exp2ax;

				for (int n = 1; ; n++)
				{
					T coef = T(expa2n2[n - 1]) * expx2 / (a2 * T(double(n) * n) + y * y);
					prod2ax *= exp2ax;
					prodm2ax *= expm2ax;

					sum1 += coef;
					sum2 += coef * prodm2ax;
					sum4 += (coef * prodm2ax) * (a * T(double(n)));
					sum3 += coef * prod2ax;
					sum5 += (coef * prod2ax) * (a * T(double(n)));

					if (get_value((coef * prod2ax) * (a * T(double(n)))) < v_relerr * get_value(sum5)) break;
				}
			}
		}
		else
		{
			T exp2ax = exp((T(2.0) * a) * x);
			T expm2ax = T(1.0) / exp2ax;

			if (vx < 5.0E-4)
			{
				T x2 = x * x;
				expx2 = T(1.0) - x2 * (T(1.0) - T(0.5) * x2);

				for (int n = 1; ; n++)
				{
					T coef = exp(-a2 * T(double(n) * n)) * expx2 / (a2 * T(double(n) * n) + y * y);
					prod2ax *= exp2ax;
					prodm2ax *= expm2ax;

					sum1 += coef;
					sum2 += coef * prodm2ax;
					sum3 += coef * prod2ax;
					sum5 += coef * (T(2.0) * a) * T(double(n)) * sinh_taylor(T(2.0) * a * T(double(n)) * x);

					if (get_value(coef * prod2ax) < v_relerr * get_value(sum3)) break;
				}
			}
			else
			{
				expx2 = exp(-x * x);
				for (int n = 1; ; n++)
				{
					T coef = exp(-a2 * T(double(n) * n)) * expx2 / (a2 * T(double(n) * n) + y * y);
					prod2ax *= exp2ax;
					prodm2ax *= expm2ax;

					sum1 += coef;
					sum2 += coef * prodm2ax;
					sum4 += (coef * prodm2ax) * (a * T(double(n)));
					sum3 += coef * prod2ax;
					sum5 += (coef * prod2ax) * (a * T(double(n)));

					if (get_value((coef * prod2ax) * (a * T(double(n)))) < v_relerr * get_value(sum5)) break;
				}
			}
		}

		T expx2erfcxy = (vya > -6.0) ? expx2 * erfcx(ya) : T(2.0) * exp(ya * ya - x * x);

		if (vya > 5.0)
		{
			T sinxy = sin(x * ya);
			ret = (expx2erfcxy - c * ya * sum1) * cos(T(2.0) * x * ya) + (c * x * expx2) * sinxy * sinc(x * ya, sinxy);
		}
		else
		{
			T xs = z.re;
			T sinxy = sin(xs * ya);
			T sin2xy = sin(T(2.0) * xs * ya);
			T cos2xy = cos(T(2.0) * xs * ya);

			T coef1 = expx2erfcxy - c * ya * sum1;
			T coef2 = c * xs * expx2;

			ret = complex<T>(coef1 * cos2xy + coef2 * sinxy * sinc(xs * ya, sinxy), coef2 * sinc(T(2.0) * xs * ya, sin2xy) - coef1 * sin2xy);
		}
	}
	else
	{
		ret = exp(-x * x);
		T n0 = floor(x / a + T(0.5));
		T dx = a * n0 - x;
		
		sum3 = exp(-dx * dx) / (a2 * (n0 * n0) + ya * ya);
		sum5 = a * n0 * sum3;

		T exp1 = exp(T(4.0) * a * dx);
		T exp1dn = T(1.0);
		int dn;

		for (dn = 1; get_value(n0) - T(double(dn)) > T(0.0); dn++)
		{
			T np = n0 + T(double(dn));
			T nm = n0 - T(double(dn));
			T tp = exp(-sqr(a * T(double(dn)) + dx));
			T tm = tp * (exp1dn *= exp1);

			tp /= (a2 * (np * np) + ya * ya);
			tm /= (a2 * (nm * nm) + ya * ya);
			sum3 += tp + tm;
			sum5 += a * (np * tp + nm * tm);

			if (get_value(a * (np * tp + nm * tm)) < v_relerr * get_value(sum5))
			{
				goto finish;
			}
		}
		while (true)
		{
			T np = n0 + T(double(dn++));
			T tp = exp(-sqr(a * T(double(dn)) + dx)) / (a2 * (np * np) + ya * ya);
			
			sum3 += tp;
			sum5 += a * np * tp;
			
			if (get_value(a * np * tp) < v_relerr * get_value(sum5))
			{
				goto finish;
			}
		}
	}
	
	finish:
	
	return ret + complex<T>((T(0.5) * c) * ya * (sum2 + sum3), (T(0.5 )* c) * copysign(sum5 - sum4, z.re));
}

}
