#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace paad::utils
{

enum class Boundary { Natural, Clamped };

class Spline
{
	private:
		void TDMA(std::vector<std::vector<double>>& A, std::vector<double>& rhs);

	public:
		std::vector<double> h = std::vector<double>(3);
		std::vector<double> a = std::vector<double>(3);
		std::vector<double> b = std::vector<double>(3);
		std::vector<double> c = std::vector<double>(3);
		std::vector<double> d = std::vector<double>(3);

		std::vector<double> x;
		std::vector<double> f;

		std::vector<Boundary> boundary = {Boundary::Natural, Boundary::Natural};

		double interpolation(double x_data);
		double derivative(double x_data);
		void makeNaturalSpline(void);
		void makeClampedSpline(void);
		void makeNaturalClampedSpline(void);
		void makeClampedNaturalSpline(void);
		void makeSpline(void);
};

inline void Spline::makeSpline(void)
{
	if(boundary[0] == Boundary::Natural && boundary[1] == Boundary::Natural) makeNaturalSpline();
	else if(boundary[0] == Boundary::Clamped && boundary[1] == Boundary::Natural) makeClampedNaturalSpline();
	else if(boundary[0] == Boundary::Natural && boundary[1] == Boundary::Clamped) makeNaturalClampedSpline();
	else if(boundary[0] == Boundary::Clamped && boundary[1] == Boundary::Clamped) makeClampedSpline();
}

inline double Spline::interpolation(double x_data)
{
	if(x_data < x[0]) return f[0];
	if(x_data >= x[x.size() - 1]) return f[x.size() - 1];

	int nx = 0;
	for(int i = 0; i < x.size() - 1; i ++)
	{
		if(x[i] <= x_data && x_data < x[i + 1]) nx = i;
	}
	if(x[x.size() - 1] == x_data) nx = x.size() - 1;

	double result = 0.0;
	result += a[nx] * (x_data - x[nx]) * (x_data - x[nx]) * (x_data - x[nx]);
	result += b[nx] * (x_data - x[nx]) * (x_data - x[nx]);
	result += c[nx] * (x_data - x[nx]);
	result += d[nx];

	return result;
}

inline double Spline::derivative(double x_data)
{
	if(x_data < x[0]) return f[0];
	if(x_data >= x[x.size() - 1]) return f[x.size() - 1];

	int nx = 0;
	for(int i = 0; i < x.size() - 1; i ++)
	{
		if(x[i] <= x_data && x_data < x[i + 1]) nx = i;
	}
	if(x[x.size() - 1] == x_data) nx = x.size() - 1;

	double result = 0.0;
	result += 3.0 * a[nx] * (x_data - x[nx]) * (x_data - x[nx]);
	result += 2.0 * b[nx] * (x_data - x[nx]);
	result += c[nx];

	return result;
}

inline void Spline::makeNaturalSpline(void)
{
	h.resize(x.size() - 1);
	a.resize(x.size() - 1);
	b.resize(x.size() - 1);
	c.resize(x.size() - 1);
	d.resize(x.size() - 1);

	std::vector<double> ypp(x.size(), 0.0);
	std::vector<double> s(x.size() - 1, 0.0);

	for(int i = 0; i < h.size(); i ++)
	{
		h[i] = x[i + 1] - x[i];
		s[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i]);
	}

	std::vector<std::vector<double>> A(x.size(), std::vector<double>(x.size(), 0.0));
	A[0][0] = 1.0;
	A[A.size() - 1][A.size() - 1] = 1.0;
	ypp[0] = 0.0;
	ypp[A.size() - 1] = 0.0;

	for(int i = 1; i < A.size() - 1; i ++)
	{
		A[i][i - 1] = h[i - 1];
		A[i][i] = 2.0 * (h[i - 1] + h[i]);
		A[i][i + 1] = h[i];
		ypp[i] = 6.0 * (s[i] - s[i - 1]);
	}

	TDMA(A, ypp);

	for(int i = 0; i < a.size(); i ++)
	{
		a[i] = (ypp[i + 1] - ypp[i]) / (6.0 * h[i]);
		b[i] = 0.5 * ypp[i];
		c[i] = s[i] - h[i] / 6.0 * (2.0 * ypp[i] + ypp[i + 1]);
		d[i] = f[i];
	}
}

inline void Spline::makeClampedSpline(void)
{
	h.resize(x.size() - 1);
	a.resize(x.size() - 1);
	b.resize(x.size() - 1);
	c.resize(x.size() - 1);
	d.resize(x.size() - 1);

	std::vector<double> ypp(x.size(), 0.0);
	std::vector<double> s(x.size() - 1, 0.0);

	for(int i = 0; i < h.size(); i ++)
	{
		h[i] = x[i + 1] - x[i];
		s[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i]);
	}

	std::vector<std::vector<double>> A(x.size(), std::vector<double>(x.size(), 0.0));
	A[0][0] = 2.0 * h[0];
	A[0][1] = h[0];
	A[A.size() - 1][A.size() - 2] = h[h.size() - 1];
	A[A.size() - 1][A.size() - 1] = 2.0 * h[h.size() - 1];
	ypp[0] = 6.0 * s[0];
	ypp[A.size() - 1] = -6.0 * s[s.size() - 1];

	for(int i = 1; i < A.size() - 1; i ++)
	{
		A[i][i - 1] = h[i - 1];
		A[i][i] = 2.0 * (h[i - 1] + h[i]);
		A[i][i + 1] = h[i];
		ypp[i] = 6.0 * (s[i] - s[i - 1]);
	}

	TDMA(A, ypp);

	for(int i = 0; i < a.size(); i ++)
	{
		a[i] = (ypp[i + 1] - ypp[i]) / (6.0 * h[i]);
		b[i] = 0.5 * ypp[i];
		c[i] = s[i] - h[i] / 6.0 * (2.0 * ypp[i] + ypp[i + 1]);
		d[i] = f[i];
	}
}

inline void Spline::makeNaturalClampedSpline(void)
{
	h.resize(x.size() - 1);
	a.resize(x.size() - 1);
	b.resize(x.size() - 1);
	c.resize(x.size() - 1);
	d.resize(x.size() - 1);

	std::vector<double> ypp(x.size(), 0.0);
	std::vector<double> s(x.size() - 1, 0.0);

	for(int i = 0; i < h.size(); i ++)
	{
		h[i] = x[i + 1] - x[i];
		s[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i]);
	}

	std::vector<std::vector<double>> A(x.size(), std::vector<double>(x.size(), 0.0));
	A[0][0] = 1.0;
	A[A.size() - 1][A.size() - 2] = h[h.size() - 1];
	A[A.size() - 1][A.size() - 1] = 2.0 * h[h.size() - 1];
	ypp[0] = 0.0;
	ypp[A.size() - 1] = -6.0 * s[s.size() - 1];

	for(int i = 1; i < A.size() - 1; i ++)
	{
		A[i][i - 1] = h[i - 1];
		A[i][i] = 2.0 * (h[i - 1] + h[i]);
		A[i][i + 1] = h[i];
		ypp[i] = 6.0 * (s[i] - s[i - 1]);
	}

	TDMA(A, ypp);

	for(int i = 0; i < a.size(); i ++)
	{
		a[i] = (ypp[i + 1] - ypp[i]) / (6.0 * h[i]);
		b[i] = 0.5 * ypp[i];
		c[i] = s[i] - h[i] / 6.0 * (2.0 * ypp[i] + ypp[i + 1]);
		d[i] = f[i];
	}
}

inline void Spline::makeClampedNaturalSpline(void)
{
	h.resize(x.size() - 1);
	a.resize(x.size() - 1);
	b.resize(x.size() - 1);
	c.resize(x.size() - 1);
	d.resize(x.size() - 1);

	std::vector<double> ypp(x.size(), 0.0);
	std::vector<double> s(x.size() - 1, 0.0);

	for(int i = 0; i < h.size(); i ++)
	{
		h[i] = x[i + 1] - x[i];
		s[i] = (f[i + 1] - f[i]) / (x[i + 1] - x[i]);
	}

	std::vector<std::vector<double>> A(x.size(), std::vector<double>(x.size(), 0.0));
	A[0][0] = 2.0 * h[0];
	A[0][1] = h[0];
	A[A.size() - 1][A.size() - 1] = 1.0;
	ypp[0] = 6.0 * s[0];
	ypp[A.size() - 1] = 0.0;

	for(int i = 1; i < A.size() - 1; i ++)
	{
		A[i][i - 1] = h[i - 1];
		A[i][i] = 2.0 * (h[i - 1] + h[i]);
		A[i][i + 1] = h[i];
		ypp[i] = 6.0 * (s[i] - s[i - 1]);
	}

	TDMA(A, ypp);

	for(int i = 0; i < a.size(); i ++)
	{
		a[i] = (ypp[i + 1] - ypp[i]) / (6.0 * h[i]);
		b[i] = 0.5 * ypp[i];
		c[i] = s[i] - h[i] / 6.0 * (2.0 * ypp[i] + ypp[i + 1]);
		d[i] = f[i];
	}
}

inline void Spline::TDMA(std::vector<std::vector<double>>& A, std::vector<double>& rhs)
{
	A[0][1] = A[0][1] / A[0][0];
	rhs[0] = rhs[0] / A[0][0];

	for(int i = 1; i < rhs.size(); i ++)
	{
		if(i < rhs.size() - 1)
		{
			A[i][i + 1] = A[i][i + 1] / (A[i][i] - A[i - 1][i] * A[i][i - 1]);
		}
		rhs[i] = (rhs[i] - rhs[i - 1] * A[i][i - 1]) / (A[i][i] - A[i - 1][i] * A[i][i - 1]);
	}

	for(int i = rhs.size() - 2; i >= 0; i --)
	{
		rhs[i] = rhs[i] - rhs[i + 1] * A[i][i + 1];
	}
}

inline double interpolate2DSpline(double x, double y, const std::vector<double>& x_grid, const std::vector<double>& y_grid, const std::vector<std::vector<double>>& f, std::vector<Boundary> boundary_x = {Boundary::Natural, Boundary::Natural}, std::vector<Boundary> boundary_y = {Boundary::Natural, Boundary::Natural})
{
	std::vector<double> fx(y_grid.size());

	for(int i = 0; i < y_grid.size(); ++i)
	{
		std::vector<double> fy(x_grid.size());
		for(int j = 0; j < x_grid.size(); ++j)
		{
			fy[j] = f[j][i];
		}

		Spline spline;
		spline.f = fy;
		spline.x = x_grid;
		spline.boundary = boundary_x;
		spline.makeSpline();
		fx[i] = spline.interpolation(x);
	}

	Spline spline;
	spline.f = fx;
	spline.x = y_grid;
	spline.boundary = boundary_y;
	spline.makeSpline();
	
	return spline.interpolation(y);
}

}
