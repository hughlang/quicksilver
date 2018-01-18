#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
/// An RGBA color represented by normalized floats
pub struct Color {
    ///The red component of the color
    pub r: f32,
    ///The green component of the color
    pub g: f32,
    ///The blue component of the color
    pub b: f32,
    ///The alpha component of the color
    pub a: f32,
}

#[allow(missing_docs)]
impl Color {
    pub fn white() -> Color {
        Color {
            r: 1f32,
            g: 1f32,
            b: 1f32,
            a: 1f32,
        }
    }

    pub fn black() -> Color {
        Color {
            r: 0f32,
            g: 0f32,
            b: 0f32,
            a: 1f32,
        }
    }

    pub fn red() -> Color {
        Color {
            r: 1f32,
            g: 0f32,
            b: 0f32,
            a: 1f32,
        }
    }

    pub fn orange() -> Color {
        Color {
            r: 1f32,
            g: 0.5f32,
            b: 0f32,
            a: 1f32,
        }
    }

    pub fn yellow() -> Color {
        Color {
            r: 1f32,
            g: 1f32,
            b: 0f32,
            a: 1f32,
        }
    }

    pub fn green() -> Color {
        Color {
            r: 0f32,
            g: 1f32,
            b: 0f32,
            a: 1f32,
        }
    }

    pub fn cyan() -> Color {
        Color {
            r: 0f32,
            g: 1f32,
            b: 1f32,
            a: 1f32,
        }
    }

    pub fn blue() -> Color {
        Color {
            r: 0f32,
            g: 0f32,
            b: 1f32,
            a: 1f32,
        }
    }

    pub fn purple() -> Color {
        Color {
            r: 1f32,
            g: 0f32,
            b: 1f32,
            a: 1f32,
        }
    }

    pub fn indigo() -> Color {
        Color {
            r: 0.5f32,
            g: 0f32,
            b: 1f32,
            a: 1f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors() {
        let colors = [Color::white(), Color::black(), Color::red(), Color::orange(), Color::yellow(),
            Color::green(), Color::cyan(), Color::blue(), Color::purple(), Color::indigo()];
        for i in 0..colors.len() {
            for j in 0..colors.len() {
                assert_eq!(i == j, colors[i].clone() == *&colors[j]);
            }
        }
    }
}
