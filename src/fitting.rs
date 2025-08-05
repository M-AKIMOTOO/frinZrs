use std::f64;
use std::error::Error; // Import Error trait
use std::fmt; // Import fmt for custom error display

// Custom error type for fitting
#[derive(Debug)]
struct FittingError(String);

impl fmt::Display for FittingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fitting Error: {}", self.0)
    }
}

impl Error for FittingError {}

// C++版の QuadraticFitResult 構造体に対応
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct QuadraticFitResult {
    pub peak_x: f64, // 2次関数の頂点のx座標
    // pub success: bool,
    pub a: f64, // y = ax^2 + bx + c の係数 a
    pub b: f64, // y = ax^2 + bx + c の係数 b
    pub c: f64, // y = ax^2 + bx + c の係数 c
    // pub message: String,
}

/// Fits a quadratic function y = ax^2 + bx + c to N points using least squares.
/// x_coords and y_values must be of the same size and contain at least 3 points.
/// Returns the x-coordinate of the vertex (-b / 2a). Success only if 'a' is negative.
pub fn fit_quadratic_least_squares(
    x_coords: &[f64],
    y_values: &[f64],
) -> Result<QuadraticFitResult, Box<dyn Error>> { // Changed return type

    let n = x_coords.len();
    let epsilon = 1e-9;

    if n < 3 || n != y_values.len() {
        return Err(Box::new(FittingError("Input vectors must be of the same size and contain at least 3 points for least squares.".to_string()))); // Changed error return
    }

    let s0 = n as f64;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;
    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let mut t2 = 0.0;

    for i in 0..n {
        let x = x_coords[i];
        let y = y_values[i];
        let x_sq = x * x;
        s1 += x;
        s2 += x_sq;
        s3 += x_sq * x;
        s4 += x_sq * x_sq;
        t0 += y;
        t1 += x * y;
        t2 += x_sq * y;
    }

    // 連立方程式を解くための行列式
    // S0*c + S1*b + S2*a = T0
    // S1*c + S2*b + S3*a = T1
    // S2*c + S3*b + S4*a = T2

    // Denominator D
    let d = s0 * (s2 * s4 - s3 * s3) -
            s1 * (s1 * s4 - s2 * s3) +
            s2 * (s1 * s3 - s2 * s2);

    if d.abs() < epsilon {
        return Err(Box::new(FittingError(format!("Denominator D ({}) is almost zero. Matrix is singular or ill-conditioned.", d)))); // Changed error return
    }

    // Numerators for c, b, a (using Cramer's rule implicitly)
    let dc_num = t0 * (s2 * s4 - s3 * s3) - s1 * (t1 * s4 - t2 * s3) + s2 * (t1 * s3 - t2 * s2);
    let db_num = s0 * (t1 * s4 - t2 * s3) - t0 * (s1 * s4 - s2 * s3) + s2 * (s1 * t2 - s2 * t1);
    let da_num = s0 * (s2 * t2 - s3 * t1) - s1 * (s1 * t2 - s2 * t1) + t0 * (s1 * s3 - s2 * s2);

    let a = da_num / d;
    let b = db_num / d;
    let c = dc_num / d;

    if a.abs() < epsilon {
        return Err(Box::new(FittingError("Coefficient 'a' is almost zero. Quadratic function is degenerate.".to_string()))); // Changed error return
    }

    if a > 0.0 {
        return Err(Box::new(FittingError("Coefficient 'a' is positive. Quadratic function is convex downwards, no maximum exists.".to_string()))); // Changed error return
    }

    let peak_x = -b / (2.0 * a);

    Ok(QuadraticFitResult {
        peak_x,
        a,
        b,
        c,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_quadratic_least_squares_basic() {
        // y = -x^2 + 2x + 3 の頂点は x = 1, y = 4
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_values = vec![3.0, 4.0, 3.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_ok());
        let fit_result = result.unwrap();

        // assert!(fit_result.success);
        assert!((fit_result.peak_x - 1.0).abs() < 1e-9);
        assert!((fit_result.a - (-1.0)).abs() < 1e-9);
        assert!((fit_result.b - 2.0).abs() < 1e-9);
        assert!((fit_result.c - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fit_quadratic_least_squares_more_points() {
        // y = -2x^2 + 8x + 1 の頂点は x = 2, y = 9
        let x_coords = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_values = vec![1.0, 7.0, 9.0, 7.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_ok());
        let fit_result = result.unwrap();

        // assert!(fit_result.success);
        assert!((fit_result.peak_x - 2.0).abs() < 1e-9);
        assert!((fit_result.a - (-2.0)).abs() < 1e-9);
        assert!((fit_result.b - 8.0).abs() < 1e-9);
        assert!((fit_result.c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fit_quadratic_least_squares_positive_a() {
        // y = x^2 の頂点は x = 0, y = 0 (下に凸)
        let x_coords = vec![-1.0, 0.0, 1.0];
        let y_values = vec![1.0, 0.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result { // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("Coefficient 'a' is positive"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }

    #[test]
    fn test_fit_quadratic_least_squares_degenerate() {
        // 直線の場合 (a=0)
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_values = vec![0.0, 1.0, 2.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result { // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("Coefficient 'a' is almost zero"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }

    #[test]
    fn test_fit_quadratic_least_squares_insufficient_points() {
        let x_coords = vec![0.0, 1.0];
        let y_values = vec![0.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result { // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("at least 3 points"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }
}
