import pandas as pd
import io

# === Load Reference Datasets ===
def load_reference_data():
    g1000 = pd.read_csv("../datasets/brca1_variants.csv", low_memory=False, on_bad_lines='skip')
    clinvar = pd.read_csv("../datasets/clinvar_brca1.csv", low_memory=False)

    # Normalize types
    g1000['CHROM'] = g1000['CHROM'].astype(str)
    g1000['POS'] = g1000['POS'].astype(int)
    g1000['REF'] = g1000['REF'].astype(str)
    g1000['ALT'] = g1000['ALT'].astype(str)

    clinvar['Chromosome'] = clinvar['Chromosome'].astype(str)
    clinvar['Start'] = clinvar['Start'].astype(int)
    clinvar['ReferenceAlleleVCF'] = clinvar['ReferenceAlleleVCF'].astype(str)
    clinvar['AlternateAlleleVCF'] = clinvar['AlternateAlleleVCF'].astype(str)

    return g1000, clinvar


# === VCF Parser ===
def parse_vcf(uploaded_file):
    variants = []

    for line in uploaded_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        chrom, pos, _, ref, alt = parts[:5]
        variants.append({
            "CHROM": chrom,
            "POS": int(pos),
            "REF": ref,
            "ALT": alt
        })

    return pd.DataFrame(variants)


# === CSV or VCF Reader ===
def load_patient_file(uploaded_file, filename: str):
    if filename.endswith(".vcf"):
        return parse_vcf(uploaded_file)
    elif filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .vcf file.")


# === Variant Analysis Logic ===
def analyze_patient_variants(patient_df, g1000_df, clinvar_df):
    report = []
    summary = {
        "pathogenic_variants": 0,
        "likely_pathogenic": 0,
        "total_matches": 0
    }

    # Normalize types
    patient_df['CHROM'] = patient_df['CHROM'].astype(str)
    patient_df['POS'] = patient_df['POS'].astype(int)
    patient_df['REF'] = patient_df['REF'].astype(str)
    patient_df['ALT'] = patient_df['ALT'].astype(str)

    # Merge with ClinVar
    merged = pd.merge(
        patient_df,
        clinvar_df,
        left_on=['CHROM', 'POS', 'REF', 'ALT'],
        right_on=['Chromosome', 'Start', 'ReferenceAlleleVCF', 'AlternateAlleleVCF'],
        how='left'
    )

    # Merge with 1000 Genomes
    g1000_merged = pd.merge(
        patient_df,
        g1000_df[['CHROM', 'POS', 'REF', 'ALT'] + list(g1000_df.columns[9:])],
        on=['CHROM', 'POS', 'REF', 'ALT'],
        how='left'
    )

    for idx, row in merged.iterrows():
        variant = f"{row['CHROM']}:{row['POS']} {row['REF']}>{row['ALT']}"

        clin_sig = row.get("ClinicalSignificance", "")
        phenotype = row.get("PhenotypeList", "Unknown condition")

        if row['POS'] in g1000_df['POS'].values:
            freq_data = g1000_merged[g1000_merged['POS'] == row['POS']]
            total_samples = len(freq_data.columns[9:])
            alt_count = (freq_data.iloc[0, 9:] != '0|0').sum()
            freq = alt_count / total_samples if total_samples else 0
        else:
            freq = 0

        if pd.notna(clin_sig):
            summary['total_matches'] += 1
            if "Pathogenic" in clin_sig:
                summary['pathogenic_variants'] += 1
            elif "Likely pathogenic" in clin_sig:
                summary['likely_pathogenic'] += 1

            report.append(
                f"⚠️ {variant} → {clin_sig} ({phenotype}) | 1000G Alt Freq: {freq:.2%}"
            )
        else:
            report.append(f"✅ {variant} → No known pathogenic record. | 1000G Alt Freq: {freq:.2%}")

    return "\n".join(report), summary
