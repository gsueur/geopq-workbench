//! Named colour maps: fixed value → colour tables for datasets whose
//! classes have an official palette.
//!
//! A categorical style normally assigns colours by frequency, which is
//! right for data whose values carry no convention. Land cover is the
//! opposite case: every CORINE class has a colour the whole field reads
//! fluently, and inventing another one makes a familiar map unreadable.
//!
//! Maps are also read from QGIS colour-map exports, so a dataset that
//! ships one can be styled by it without anything being added here.

/// One class: the attribute value, its colour, and the label to show in
/// the legend instead of the bare code.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassColor {
    pub value: String,
    pub rgb: [u8; 3],
    pub label: String,
}

/// A named table, applied when the styled column's values match it.
#[derive(Clone, Debug)]
pub struct ColorMap {
    pub name: String,
    pub classes: Vec<ClassColor>,
    /// Column names that carry this map's codes. A dataset's own column
    /// naming identifies it before a single row is read, which matters:
    /// the class list comes from the map, so recognizing it by name
    /// means the style applies with no scan at all.
    pub columns: Vec<String>,
}

impl ColorMap {
    pub fn get(&self, value: &str) -> Option<&ClassColor> {
        self.classes.iter().find(|c| c.value == value)
    }

    /// Share of `values` this map can colour. The style dialog offers a
    /// map only when it covers essentially everything, since a partial
    /// match would leave a scatter of features in the "other" colour.
    pub fn coverage(&self, values: &[String]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let hit = values.iter().filter(|v| self.get(v).is_some()).count();
        hit as f64 / values.len() as f64
    }
}

/// CORINE Land Cover, the official palette (CLC 2018 nomenclature).
/// Values are the `CODE_18` / `Code_18` three-digit class codes.
const CLC: [(&str, [u8; 3], &str); 47] = [
    ("111", [230, 0, 77], "Continuous urban fabric"),
    ("112", [255, 0, 0], "Discontinuous urban fabric"),
    ("121", [204, 77, 242], "Industrial or commercial units"),
    ("122", [204, 0, 0], "Road and rail networks and associated land"),
    ("123", [230, 204, 204], "Port areas"),
    ("124", [230, 204, 230], "Airports"),
    ("131", [166, 0, 204], "Mineral extraction sites"),
    ("132", [166, 77, 0], "Dump sites"),
    ("133", [255, 77, 255], "Construction sites"),
    ("141", [255, 166, 255], "Green urban areas"),
    ("142", [255, 230, 255], "Sport and leisure facilities"),
    ("211", [255, 255, 168], "Non-irrigated arable land"),
    ("212", [255, 255, 0], "Permanently irrigated land"),
    ("213", [230, 230, 0], "Rice fields"),
    ("221", [230, 128, 0], "Vineyards"),
    ("222", [242, 166, 77], "Fruit trees and berry plantations"),
    ("223", [230, 166, 0], "Olive groves"),
    ("231", [230, 230, 77], "Pastures"),
    ("241", [255, 230, 166], "Annual crops associated with permanent crops"),
    ("242", [255, 230, 77], "Complex cultivation patterns"),
    ("243", [230, 204, 77], "Land principally occupied by agriculture with significant areas of natural vegetation"),
    ("244", [242, 204, 166], "Agro-forestry areas"),
    ("311", [128, 255, 0], "Broad-leaved forest"),
    ("312", [0, 166, 0], "Coniferous forest"),
    ("313", [77, 255, 0], "Mixed forest"),
    ("321", [204, 242, 77], "Natural grasslands"),
    ("322", [166, 255, 128], "Moors and heathland"),
    ("323", [166, 230, 77], "Sclerophyllous vegetation"),
    ("324", [166, 242, 0], "Transitional woodland-shrub"),
    ("331", [230, 230, 230], "Beaches - dunes - sands"),
    ("332", [204, 204, 204], "Bare rocks"),
    ("333", [204, 255, 204], "Sparsely vegetated areas"),
    ("334", [0, 0, 0], "Burnt areas"),
    ("335", [166, 230, 204], "Glaciers and perpetual snow"),
    ("411", [166, 166, 255], "Inland marshes"),
    ("412", [77, 77, 255], "Peat bogs"),
    ("421", [204, 204, 255], "Salt marshes"),
    ("422", [230, 230, 255], "Salines"),
    ("423", [166, 166, 230], "Intertidal flats"),
    ("511", [0, 204, 242], "Water courses"),
    ("512", [128, 242, 230], "Water bodies"),
    ("521", [0, 255, 166], "Coastal lagoons"),
    ("522", [166, 255, 230], "Estuaries"),
    ("523", [230, 242, 255], "Sea and ocean"),
    ("999", [255, 255, 255], "NODATA"),
    ("990", [255, 255, 255], "UNCLASSIFIED LAND SURFACE"),
    ("995", [230, 242, 255], "UNCLASSIFIED WATER BODIES"),
];

/// Every built-in map.
pub fn builtins() -> Vec<ColorMap> {
    vec![ColorMap {
        name: "CORINE Land Cover".to_string(),
        // CLC ships its class code as CODE_<year> (Code_18, CODE_12,
        // code_06 …); the EEA's own vector products use these names.
        columns: ["code_18", "code_12", "code_06", "code_00", "code_90", "clc_code"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        classes: CLC
            .iter()
            .map(|(v, rgb, l)| ClassColor {
                value: (*v).to_string(),
                rgb: *rgb,
                label: (*l).to_string(),
            })
            .collect(),
    }]
}

/// The built-in map a column of this name carries, if any. Matched
/// case-insensitively: the same product ships `Code_18` and `CODE_18`.
pub fn match_column(column: &str) -> Option<ColorMap> {
    builtins()
        .into_iter()
        .find(|m| m.columns.iter().any(|c| c.eq_ignore_ascii_case(column)))
}

/// The built-in map that covers `values`, if one does.
///
/// The threshold is deliberately high: CORINE colours applied to a
/// column that merely looks like class codes would be a confident lie.
pub fn match_builtin(values: &[String]) -> Option<ColorMap> {
    builtins()
        .into_iter()
        .find(|m| m.coverage(values) >= 0.9)
}

/// Parse a QGIS colour-map export.
///
/// ```text
/// # QGIS Generated Color Map Export File
/// INTERPOLATION:DISCRETE
/// 1,230,000,077,255, 111 - Continuous urban fabric
/// ```
///
/// The leading field is the raster value and the trailing text is the
/// label, which for a vector join is where the class value actually
/// lives ("111 - Continuous urban fabric"). Both readings are kept: the
/// label's leading code when it has one, the raster value otherwise.
pub fn parse_qgis(text: &str, name: &str) -> Option<ColorMap> {
    let mut classes: Vec<ClassColor> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("INTERPOLATION") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 5 {
            continue;
        }
        let byte = |i: usize| parts.get(i).and_then(|p| p.parse::<u8>().ok());
        let (Some(r), Some(g), Some(b)) = (byte(1), byte(2), byte(3)) else {
            continue;
        };
        // Everything past the alpha channel is the label, commas and all.
        let rest = parts[5..].join(",");
        let rest = rest.trim();
        let (value, label) = match rest.split_once('-') {
            Some((code, label)) if !code.trim().is_empty() => {
                (code.trim().to_string(), label.trim().to_string())
            }
            _ => (parts[0].to_string(), rest.to_string()),
        };
        if value.is_empty() || classes.iter().any(|c| c.value == value) {
            continue;
        }
        classes.push(ClassColor {
            value,
            rgb: [r, g, b],
            label: if label.is_empty() { String::new() } else { label },
        });
    }
    (!classes.is_empty()).then_some(ColorMap {
        name: name.to_string(),
        classes,
        columns: Vec::new(),
    })
}

/// Categorical style for `values`, using `map` when one was recognized
/// and kept.
///
/// The map also fixes the order: frequency order scatters the
/// nomenclature, and a CORINE legend reading 111, 112, 121 … is half the
/// point of having one.
pub fn categorical_mode(
    values: Vec<String>,
    map: Option<&ColorMap>,
) -> crate::data::layer::StyleMode {
    use crate::data::layer::{StyleMode, STYLE_BINS};
    let Some(map) = map else {
        return StyleMode::Categorical {
            values,
            colors: None,
            labels: None,
        };
    };
    // With no scanned values the map itself is the class list — which is
    // the point: a published nomenclature does not need to be
    // rediscovered by reading the data.
    let mut ordered: Vec<&ClassColor> = map
        .classes
        .iter()
        .filter(|c| values.is_empty() || values.iter().any(|v| *v == c.value))
        .collect();
    // Values the map does not know keep their place at the end; they
    // fall into "other" beyond the bin ceiling anyway.
    ordered.truncate(STYLE_BINS - 1);
    StyleMode::Categorical {
        values: ordered.iter().map(|c| c.value.clone()).collect(),
        colors: Some(ordered.iter().map(|c| c.rgb).collect()),
        labels: Some(
            ordered
                .iter()
                .map(|c| {
                    if c.label.is_empty() {
                        c.value.clone()
                    } else {
                        format!("{} — {}", c.value, c.label)
                    }
                })
                .collect(),
        ),
    }
}

/// The style a dataset announces through its own schema, if any:
/// a text column named for a built-in nomenclature, styled by it.
///
/// Applied at load so an authoritative classification arrives drawn in
/// its own colours instead of a flat fill the user has to replace by
/// hand. It costs nothing to be sure of: the column name decides, the
/// class list comes from the map, and the geometry is binned on its
/// first build like any other style.
///
/// Only non-numeric columns qualify. A code column typed as an integer
/// is styled as a graduated ramp everywhere else in the app, and a
/// categorical style over it would classify values the renderer bins as
/// numbers.
pub fn schema_style(columns: &[(String, bool)]) -> Option<crate::data::layer::StyleBy> {
    let (column, map) = columns
        .iter()
        .filter(|(_, numeric)| !*numeric)
        .find_map(|(name, _)| match_column(name).map(|m| (name.clone(), m)))?;
    Some(crate::data::layer::StyleBy {
        mode: categorical_mode(Vec::new(), Some(&map)),
        column,
        ramp: crate::data::layer::Ramp::Viridis, // unused: the map holds the colours
        hidden_bins: 0,
        per_area: false,
        classified_rows: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clc_table_is_the_official_palette() {
        let m = &builtins()[0];
        // Spot values every CORINE user knows by sight.
        assert_eq!(m.get("111").unwrap().rgb, [230, 0, 77], "continuous urban");
        assert_eq!(m.get("512").unwrap().rgb, [128, 242, 230], "water bodies");
        assert_eq!(m.get("311").unwrap().label, "Broad-leaved forest");
        // Codes are unique, and the nomenclature's five top levels are
        // all represented.
        for lead in ['1', '2', '3', '4', '5'] {
            assert!(
                m.classes.iter().any(|c| c.value.starts_with(lead)),
                "no class in level {lead}"
            );
        }
    }

    #[test]
    fn the_column_name_alone_identifies_a_map() {
        // The point of this path: no values needed, so no scan.
        for c in ["Code_18", "CODE_18", "code_12", "clc_code"] {
            assert_eq!(
                match_column(c).map(|m| m.name),
                Some("CORINE Land Cover".to_string()),
                "{c}"
            );
        }
        assert!(match_column("code").is_none());
        assert!(match_column("shapeName").is_none());
        assert!(match_column("").is_none());
    }

    #[test]
    fn a_schema_carrying_a_nomenclature_styles_itself() {
        let cols = vec![
            ("Area_Ha".to_string(), true),
            ("Code_18".to_string(), false),
            ("Remark".to_string(), false),
        ];
        let sb = super::schema_style(&cols).expect("CLC schema");
        assert_eq!(sb.column, "Code_18");
        match &sb.mode {
            crate::data::layer::StyleMode::Categorical { values, colors, labels } => {
                assert_eq!(values.len(), 47, "every class, none read from the data");
                assert_eq!(values[0], "111");
                assert_eq!(colors.as_ref().unwrap()[0], [230, 0, 77]);
                assert!(labels.as_ref().unwrap()[0].contains("Continuous urban"));
            }
            m => panic!("expected categorical, got {m:?}"),
        }
        // Nothing to recognize.
        assert!(super::schema_style(&[("shapeName".to_string(), false)]).is_none());
        assert!(super::schema_style(&[]).is_none());
        // The same name typed as a number is a graduated column
        // everywhere else; leave it alone rather than bin codes as values.
        assert!(super::schema_style(&[("code_18".to_string(), true)]).is_none());
    }

    #[test]
    fn a_map_is_offered_only_when_it_covers_the_data() {
        let clc: Vec<String> = ["111", "243", "512", "324"].iter().map(|s| s.to_string()).collect();
        assert_eq!(match_builtin(&clc).unwrap().name, "CORINE Land Cover");
        // One stray value among real ones is still CORINE.
        let real = ["112", "211", "231", "242", "311", "312", "313", "321", "324", "511"];
        let mut mostly: Vec<String> = real.iter().map(|s| s.to_string()).collect();
        mostly.push("not-a-code".into());
        assert!(match_builtin(&mostly).is_some(), "10 of 11 codes match");
        // Half unknown is not a match: those features would all land in
        // the "other" colour and the map would lie about the rest.
        let half: Vec<String> = real
            .iter()
            .map(|s| s.to_string())
            .chain((0..10).map(|i| format!("zz{i}")))
            .collect();
        assert!(match_builtin(&half).is_none());
        // A column of unrelated codes is not.
        let other: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
        assert!(match_builtin(&other).is_none());
        assert!(match_builtin(&[]).is_none());
    }

    #[test]
    fn qgis_export_parses_including_ragged_spacing() {
        // Lines 9 and 10 of the shipped CORINE legend have the spaces in
        // different places; the file is still the source of truth.
        let text = "# QGIS Generated Color Map Export File\n\
                    INTERPOLATION:DISCRETE\n\
                    1,230,000,077,255, 111 - Continuous urban fabric\n\
                    9,255,077,255, 255,133 - Construction sites\n\
                    \n\
                    44,230,242,255,255,523 - Sea and ocean\n";
        let m = parse_qgis(text, "test").unwrap();
        assert_eq!(m.classes.len(), 3);
        assert_eq!(m.get("111").unwrap().rgb, [230, 0, 77]);
        assert_eq!(m.get("133").unwrap().rgb, [255, 77, 255]);
        assert_eq!(m.get("133").unwrap().label, "Construction sites");
        assert_eq!(m.get("523").unwrap().label, "Sea and ocean");
        // A label with no code falls back to the raster value.
        let m = parse_qgis("7,1,2,3,255,Just a label", "t").unwrap();
        assert_eq!(m.classes[0].value, "7");
        assert_eq!(m.classes[0].label, "Just a label");
        // Junk yields nothing rather than an empty map.
        assert!(parse_qgis("nonsense", "t").is_none());
        assert!(parse_qgis("", "t").is_none());
    }
}
