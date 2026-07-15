//! Curated official national CRSs.
//!
//! There is no authoritative machine-readable registry of "the national
//! grid" per country — the EPSG database has areas-of-use but no national
//! flag. This table follows the community-curated Wikipedia "List of
//! national coordinate reference systems", restricted to entries that
//! resolve in the embedded EPSG database and whose projection type proj4rs
//! supports (validated by the test below).

/// One national CRS with its lon/lat area of applicability.
pub struct NationalCrs {
    pub country: &'static str,
    pub name: &'static str,
    pub epsg: u32,
    /// [lon_min, lat_min, lon_max, lat_max] of the mainland area of use.
    pub bbox: [f64; 4],
}

pub static NATIONAL_CRS: &[NationalCrs] = &[
    NationalCrs { country: "Australia", name: "GDA94 / Australian Albers", epsg: 3577, bbox: [112.0, -44.0, 154.0, -9.0] },
    NationalCrs { country: "Austria", name: "MGI / Austria Lambert", epsg: 31287, bbox: [9.5, 46.3, 17.2, 49.1] },
    NationalCrs { country: "Belgium", name: "BD72 / Belgian Lambert 72", epsg: 31370, bbox: [2.5, 49.4, 6.5, 51.6] },
    NationalCrs { country: "Canada", name: "NAD83 / Canada Atlas Lambert", epsg: 3978, bbox: [-141.0, 41.6, -52.5, 84.0] },
    NationalCrs { country: "Croatia", name: "HTRS96 / Croatia TM", epsg: 3765, bbox: [13.4, 42.3, 19.5, 46.6] },
    NationalCrs { country: "Czechia", name: "S-JTSK / Krovak East North", epsg: 5514, bbox: [12.0, 48.5, 18.9, 51.1] },
    NationalCrs { country: "Denmark", name: "ETRS89 / UTM 32N", epsg: 25832, bbox: [8.0, 54.5, 12.8, 57.8] },
    NationalCrs { country: "Estonia", name: "Estonian Coordinate System 1997", epsg: 3301, bbox: [21.7, 57.5, 28.2, 59.7] },
    NationalCrs { country: "Finland", name: "ETRS89 / TM35FIN", epsg: 3067, bbox: [19.3, 59.7, 31.6, 70.1] },
    NationalCrs { country: "France", name: "RGF93 / Lambert-93", epsg: 2154, bbox: [-5.5, 41.0, 10.0, 51.5] },
    NationalCrs { country: "Germany", name: "ETRS89 / UTM 32N", epsg: 25832, bbox: [5.8, 47.2, 15.1, 55.1] },
    NationalCrs { country: "Greece", name: "GGRS87 / Greek Grid", epsg: 2100, bbox: [19.5, 34.8, 28.3, 41.8] },
    NationalCrs { country: "Hungary", name: "HD72 / EOV", epsg: 23700, bbox: [16.1, 45.7, 22.9, 48.6] },
    NationalCrs { country: "Iceland", name: "ISN93 / Lambert 1993", epsg: 3057, bbox: [-24.7, 63.2, -13.4, 66.6] },
    NationalCrs { country: "Ireland", name: "IRENET95 / Irish Transverse Mercator", epsg: 2157, bbox: [-10.9, 51.4, -5.3, 55.5] },
    NationalCrs { country: "Lithuania", name: "LKS94 / Lithuania TM", epsg: 3346, bbox: [20.9, 53.9, 26.9, 56.5] },
    NationalCrs { country: "Luxembourg", name: "LUREF / Luxembourg TM", epsg: 2169, bbox: [5.7, 49.4, 6.6, 50.2] },
    NationalCrs { country: "Netherlands", name: "Amersfoort / RD New", epsg: 28992, bbox: [3.2, 50.7, 7.3, 53.6] },
    NationalCrs { country: "New Zealand", name: "NZGD2000 / NZTM2000", epsg: 2193, bbox: [166.0, -47.5, 178.6, -34.0] },
    NationalCrs { country: "Norway", name: "ETRS89 / UTM 33N", epsg: 25833, bbox: [4.5, 57.9, 31.2, 71.2] },
    NationalCrs { country: "Poland", name: "ETRS89 / Poland CS92", epsg: 2180, bbox: [14.1, 49.0, 24.2, 55.0] },
    NationalCrs { country: "Portugal", name: "ETRS89 / Portugal TM06", epsg: 3763, bbox: [-9.6, 36.9, -6.1, 42.2] },
    NationalCrs { country: "Romania", name: "Pulkovo 1942(58) / Stereo70", epsg: 3844, bbox: [20.2, 43.6, 29.7, 48.3] },
    NationalCrs { country: "Slovenia", name: "D96 / Slovenia TM", epsg: 3794, bbox: [13.3, 45.4, 16.6, 46.9] },
    NationalCrs { country: "South Korea", name: "Korea 2000 / Unified CS", epsg: 5179, bbox: [124.5, 33.0, 131.0, 38.7] },
    NationalCrs { country: "Spain", name: "ETRS89 / UTM 30N", epsg: 25830, bbox: [-7.5, 35.9, 0.0, 43.8] },
    NationalCrs { country: "Sweden", name: "SWEREF99 TM", epsg: 3006, bbox: [11.0, 55.2, 24.2, 69.1] },
    NationalCrs { country: "Switzerland", name: "CH1903+ / LV95", epsg: 2056, bbox: [5.9, 45.8, 10.5, 47.9] },
    NationalCrs { country: "United Kingdom", name: "OSGB36 / British National Grid", epsg: 27700, bbox: [-8.7, 49.8, 1.8, 60.9] },
    NationalCrs { country: "USA (CONUS)", name: "NAD83 / Conus Albers", epsg: 5070, bbox: [-125.0, 24.0, -66.5, 49.5] },
];

/// The national CRS whose area of use contains the given lon/lat extent
/// (with tolerance), preferring the tightest-fitting country when several
/// match (Luxembourg over a Europe-wide candidate, etc.).
pub fn national_for_extent(bbox: [f64; 4]) -> Option<&'static NationalCrs> {
    let (cx, cy) = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5);
    NATIONAL_CRS
        .iter()
        .filter(|n| {
            let b = n.bbox;
            // Center inside, extent inside a 20%-inflated area of use.
            let (mx, my) = ((b[2] - b[0]) * 0.2, (b[3] - b[1]) * 0.2);
            cx >= b[0]
                && cx <= b[2]
                && cy >= b[1]
                && cy <= b[3]
                && bbox[0] >= b[0] - mx
                && bbox[1] >= b[1] - my
                && bbox[2] <= b[2] + mx
                && bbox[3] <= b[3] + my
        })
        .min_by(|a, b| {
            let area = |n: &NationalCrs| (n.bbox[2] - n.bbox[0]) * (n.bbox[3] - n.bbox[1]);
            area(a).partial_cmp(&area(b)).unwrap()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::crs::{transform_point, Crs};

    /// Every table entry must resolve in the embedded EPSG database, parse
    /// in proj4rs, and roundtrip its own bbox center.
    #[test]
    fn all_national_crs_resolve_and_roundtrip() {
        for n in NATIONAL_CRS {
            let crs = Crs::from_epsg(n.epsg)
                .unwrap_or_else(|e| panic!("{} EPSG:{}: {e}", n.country, n.epsg));
            assert!(!crs.is_latlong, "{}: national grid must be projected", n.country);
            let (cx, cy) = ((n.bbox[0] + n.bbox[2]) * 0.5, (n.bbox[1] + n.bbox[3]) * 0.5);
            let (x, y) = transform_point(&Crs::wgs84(), &crs, cx, cy)
                .unwrap_or_else(|e| panic!("{}: forward: {e}", n.country));
            assert!(x.is_finite() && y.is_finite(), "{}", n.country);
            let (lon, lat) = transform_point(&crs, &Crs::wgs84(), x, y).unwrap();
            assert!(
                (lon - cx).abs() < 1e-6 && (lat - cy).abs() < 1e-6,
                "{}: roundtrip drift",
                n.country
            );
        }
    }

    #[test]
    fn extent_matching_prefers_tightest_country() {
        // France-extent data → Lambert-93.
        let n = national_for_extent([-4.5, 42.0, 8.0, 51.0]).unwrap();
        assert_eq!(n.epsg, 2154);
        // Luxembourg data matches Luxembourg, not Belgium/Germany.
        let n = national_for_extent([5.8, 49.5, 6.5, 50.1]).unwrap();
        assert_eq!(n.epsg, 2169);
        // Massachusetts data → CONUS Albers.
        let n = national_for_extent([-73.5, 41.2, -69.9, 42.9]).unwrap();
        assert_eq!(n.epsg, 5070);
        // Mid-ocean → nothing.
        assert!(national_for_extent([-40.0, 30.0, -35.0, 35.0]).is_none());
    }
}
