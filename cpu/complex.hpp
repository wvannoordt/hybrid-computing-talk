#pragma once

struct complex_t
{
	double x, y;
	complex_t(){}
	complex_t (const double& val) : x{val}, y{0.0} {}
	complex_t (const double& xx, const double& yy) : x{xx}, y{yy} {}
	
	complex_t operator ~ () const
	{
		return complex_t(x, -y);
	}
	
	complex_t operator + (const complex_t& rhs) const
	{
		return complex_t(x + rhs.x, y + rhs.y);
	}
	
	complex_t operator - (const complex_t& rhs) const
	{
		return complex_t(x - rhs.x, y - rhs.y);
	}
	
	complex_t operator * (const complex_t& rhs) const
	{
		return complex_t(x*rhs.x - y*rhs.y, x*rhs.y + y*rhs.x);
	}
	
	complex_t operator / (const complex_t& rhs) const
	{
		const auto denom = rhs.x*rhs.x + rhs.y*rhs.y;
		return complex_t ((x*rhs.x + y*rhs.y) / denom, (y*rhs.x - x*rhs.y) / denom);
	}
};

complex_t operator + (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) + rhs;
}

complex_t operator - (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) - rhs;
}

complex_t operator * (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) * rhs;
}

complex_t operator / (const double& dval, const complex_t& rhs)
{
	return complex_t(dval) / rhs;
}