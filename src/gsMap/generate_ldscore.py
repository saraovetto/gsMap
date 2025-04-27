"""
Module for generating LD scores for each spot in spatial transcriptomics data.

This module is responsible for assigning gene specificity scores to SNPs
and computing stratified LD scores that will be used for spatial LDSC analysis.
"""

import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import zarr
from scipy.sparse import csr_matrix
from tqdm import trange

from gsMap.config import GenerateLDScoreConfig
from gsMap.utils.generate_r2_matrix import getBlockLefts, load_bfile

# Configure warning behavior more precisely
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
logger = logging.getLogger(__name__)


def load_gtf(
    gtf_file: str, mk_score: pd.DataFrame, window_size: int
) -> tuple[pr.PyRanges, pd.DataFrame]:
    """
    Load and process the gene annotation file (GTF).

    Parameters
    ----------
    gtf_file : str
        Path to the GTF file
    mk_score : pd.DataFrame
        DataFrame containing marker scores
    window_size : int
        Window size around gene bodies in base pairs

    Returns
    -------
    tuple
        A tuple containing (gtf_pr, mk_score) where:
        - gtf_pr is a PyRanges object with gene coordinates
        - mk_score is the filtered marker score DataFrame
    """
    logger.info("Loading GTF data from %s", gtf_file)

    # Load GTF file
    gtf = pr.read_gtf(gtf_file)
    gtf = gtf.df

    # Filter for gene features
    gtf = gtf[gtf["Feature"] == "gene"]

    # Find common genes between GTF and marker scores
    # common_gene = np.intersect1d(mk_score.index, gtf.gene_name)
    common_gene = list(set(mk_score.index) & set(gtf.gene_name))
    logger.info(f"Found {len(common_gene)} common genes between GTF and marker scores")

    # Filter GTF and marker scores to common genes
    gtf = gtf[gtf.gene_name.isin(common_gene)]
    mk_score = mk_score[mk_score.index.isin(common_gene)]

    # Remove duplicated gene entries
    gtf = gtf.drop_duplicates(subset="gene_name", keep="first")

    # Process the GTF (open window around gene coordinates)
    gtf_bed = gtf[["Chromosome", "Start", "End", "gene_name", "Strand"]].copy()
    gtf_bed["Chromosome"] = gtf_bed["Chromosome"].apply(
        lambda x: f"chr{x}" if not str(x).startswith("chr") else x
    )
    gtf_bed.loc[:, "TSS"] = gtf_bed["Start"]
    gtf_bed.loc[:, "TED"] = gtf_bed["End"]

    # Create windows around genes
    gtf_bed.loc[:, "Start"] = gtf_bed["TSS"] - window_size
    gtf_bed.loc[:, "End"] = gtf_bed["TED"] + window_size
    gtf_bed.loc[gtf_bed["Start"] < 0, "Start"] = 0

    # Handle genes on negative strand (swap TSS and TED)
    tss_neg = gtf_bed.loc[gtf_bed["Strand"] == "-", "TSS"]
    ted_neg = gtf_bed.loc[gtf_bed["Strand"] == "-", "TED"]
    gtf_bed.loc[gtf_bed["Strand"] == "-", "TSS"] = ted_neg
    gtf_bed.loc[gtf_bed["Strand"] == "-", "TED"] = tss_neg
    gtf_bed = gtf_bed.drop("Strand", axis=1)

    # Convert to PyRanges
    gtf_pr = pr.PyRanges(gtf_bed)

    return gtf_pr, mk_score


def load_marker_score(mk_score_file: str) -> pd.DataFrame:
    """
    Load marker scores from a feather file.

    Parameters
    ----------
    mk_score_file : str
        Path to the marker score feather file

    Returns
    -------
    pd.DataFrame
        DataFrame with marker scores indexed by gene names
    """
    mk_score = pd.read_feather(mk_score_file).set_index("HUMAN_GENE_SYM").rename_axis("gene_name")
    mk_score = mk_score.astype(np.float32, copy=False)
    return mk_score


def load_bim(bfile_root: str, chrom: int) -> tuple[pd.DataFrame, pr.PyRanges]:
    """
    Load PLINK BIM file and convert to a PyRanges object.

    Parameters
    ----------
    bfile_root : str
        Root path for PLINK bfiles
    chrom : int
        Chromosome number

    Returns
    -------
    tuple
        A tuple containing (bim_df, bim_pr) where:
        - bim_df is a pandas DataFrame with BIM data
        - bim_pr is a PyRanges object with BIM data
    """
    bim_file = f"{bfile_root}.{chrom}.bim"
    logger.info(f"Loading BIM file: {bim_file}")

    bim = pd.read_csv(bim_file, sep="\t", header=None)
    bim.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]

    # Convert to PyRanges
    bim_pr = bim.copy()
    bim_pr.columns = ["Chromosome", "SNP", "CM", "Start", "A1", "A2"]

    # Adjust coordinates (BIM is 1-based, PyRanges uses 0-based)
    bim_pr["End"] = bim_pr["Start"].copy()
    bim_pr["Start"] = bim_pr["Start"] - 1

    bim_pr = pr.PyRanges(bim_pr)
    bim_pr.Chromosome = f"chr{chrom}"

    return bim, bim_pr


def overlaps_gtf_bim(gtf_pr: pr.PyRanges, bim_pr: pr.PyRanges) -> pd.DataFrame:
    """
    Find overlaps between GTF and BIM data, and select nearest gene for each SNP.

    Parameters
    ----------
    gtf_pr : pr.PyRanges
        PyRanges object with gene coordinates
    bim_pr : pr.PyRanges
        PyRanges object with SNP coordinates

    Returns
    -------
    pd.DataFrame
        DataFrame with SNP-gene pairs where each SNP is matched to its closest gene
    """
    # Join the PyRanges objects to find overlaps
    overlaps = gtf_pr.join(bim_pr)
    overlaps = overlaps.df

    # Calculate distance to TSS
    overlaps["Distance"] = np.abs(overlaps["Start_b"] - overlaps["TSS"])

    # For each SNP, select the closest gene
    nearest_genes = overlaps.loc[overlaps.groupby("SNP").Distance.idxmin()]

    return nearest_genes


def filter_snps_by_keep_snp(bim_df: pd.DataFrame, keep_snp_file: str) -> pd.DataFrame:
    """
    Filter BIM DataFrame to keep only SNPs in a provided list.

    Parameters
    ----------
    bim_df : pd.DataFrame
        DataFrame with BIM data
    keep_snp_file : str
        Path to a file with SNP IDs to keep

    Returns
    -------
    pd.DataFrame
        Filtered BIM DataFrame
    """
    # Read SNPs to keep
    keep_snp = pd.read_csv(keep_snp_file, header=None)[0].to_list()

    # Filter the BIM DataFrame
    filtered_bim_df = bim_df[bim_df["SNP"].isin(keep_snp)]

    logger.info(f"Kept {len(filtered_bim_df)} SNPs out of {len(bim_df)} after filtering")

    return filtered_bim_df


def get_snp_counts(config: GenerateLDScoreConfig) -> dict:
    """
    Count SNPs per chromosome and calculate start positions for zarr arrays.

    Parameters
    ----------
    config : GenerateLDScoreConfig
        Configuration object

    Returns
    -------
    dict
        Dictionary with SNP counts and start positions
    """
    snp_counts = {}
    total_snp = 0

    for chrom in range(1, 23):
        bim_df, _ = load_bim(config.bfile_root, chrom)

        if config.keep_snp_root:
            keep_snp_file = f"{config.keep_snp_root}.{chrom}.snp"
            filtered_bim_df = filter_snps_by_keep_snp(bim_df, keep_snp_file)
        else:
            filtered_bim_df = bim_df

        snp_counts[chrom] = filtered_bim_df.shape[0]
        total_snp += snp_counts[chrom]

    snp_counts["total"] = total_snp

    # Calculate cumulative SNP counts for zarr array indexing
    chrom_snp_length_array = np.array([snp_counts[chrom] for chrom in range(1, 23)]).cumsum()
    snp_counts["chrom_snp_start_point"] = [0] + chrom_snp_length_array.tolist()

    return snp_counts


def get_snp_pass_maf(bfile_root: str, chrom: int, maf_min: float = 0.05) -> list[str]:
    """
    Get SNPs that pass the minimum minor allele frequency (MAF) threshold.

    Parameters
    ----------
    bfile_root : str
        Root path for PLINK bfiles
    chrom : int
        Chromosome number
    maf_min : float, optional
        Minimum MAF threshold, by default 0.05

    Returns
    -------
    list
        List of SNP IDs that pass the MAF threshold
    """
    array_snps, array_indivs, geno_array = load_bfile(
        bfile_chr_prefix=f"{bfile_root}.{chrom}", mafMin=maf_min
    )

    m = len(array_snps.IDList)
    n = len(array_indivs.IDList)
    logger.info(
        f"Loading genotype data for {m} SNPs and {n} individuals from {bfile_root}.{chrom}"
    )

    # Filter SNPs by MAF
    snp_pass_maf = array_snps.IDList.iloc[geno_array.kept_snps]
    logger.info(f"After filtering SNPs with MAF < {maf_min}, {len(snp_pass_maf)} SNPs remain")

    return snp_pass_maf.SNP.to_list()


def get_ldscore(
    bfile_root: str,
    chrom: int,
    annot_matrix: np.ndarray,
    ld_wind: float,
    ld_unit: str = "CM",
    keep_snps_index: list[int] = None,
) -> pd.DataFrame:
    """
    Calculate LD scores using PLINK data and an annotation matrix.

    Parameters
    ----------
    bfile_root : str
        Root path for PLINK bfiles
    chrom : int
        Chromosome number
    annot_matrix : np.ndarray
        Annotation matrix
    ld_wind : float
        LD window size
    ld_unit : str, optional
        Unit for the LD window, by default "CM"
    keep_snps_index : list[int], optional
        Indices of SNPs to keep, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated LD scores
    """
    array_snps, array_indivs, geno_array = load_bfile(
        bfile_chr_prefix=f"{bfile_root}.{chrom}", keep_snps=keep_snps_index
    )

    annot_matrix = annot_matrix[geno_array.kept_snps, :]

    # Configure LD window based on specified unit
    if ld_unit == "SNP":
        max_dist = ld_wind
        coords = np.array(range(geno_array.m))
    elif ld_unit == "KB":
        max_dist = ld_wind * 1000
        coords = np.array(array_snps.df["BP"])[geno_array.kept_snps]
    elif ld_unit == "CM":
        max_dist = ld_wind
        coords = np.array(array_snps.df["CM"])[geno_array.kept_snps]
        # Check if the CM is all 0
        if np.all(coords == 0):
            logger.warning(
                "All CM values are 0 in the BIM file. Using 1MB window size for LD score calculation."
            )
            max_dist = 1_000_000
            coords = np.array(array_snps.df["BP"])[geno_array.kept_snps]
    else:
        raise ValueError(f"Invalid ld_wind_unit: {ld_unit}. Must be one of: SNP, KB, CM")

    # Calculate blocks for LD computation
    block_left = getBlockLefts(coords, max_dist)
    assert block_left.sum() > 0, "Invalid window size, please check the ld_wind parameter."

    # Calculate LD scores
    ld_scores = pd.DataFrame(geno_array.ldScoreVarBlocks(block_left, 100, annot=annot_matrix))

    return ld_scores


def calculate_ldscore_from_annotation(
    snp_annotation_df: pd.DataFrame,
    chrom: int,
    bfile_root: str,
    ld_wind: float = 1,
    ld_unit: str = "CM",
) -> pd.DataFrame:
    """
    Calculate LD scores from SNP annotation DataFrame.

    Parameters
    ----------
    snp_annotation_df : pd.DataFrame
        DataFrame with SNP annotations
    chrom : int
        Chromosome number
    bfile_root : str
        Root path for PLINK bfiles
    ld_wind : float, optional
        LD window size, by default 1
    ld_unit : str, optional
        Unit for the LD window, by default "CM"

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated LD scores
    """
    # Calculate LD scores
    snp_gene_weight_matrix = get_ldscore(
        bfile_root, chrom, snp_annotation_df.values, ld_wind=ld_wind, ld_unit=ld_unit
    )

    # Set proper data types and indices
    snp_gene_weight_matrix = snp_gene_weight_matrix.astype(np.float32, copy=False)
    snp_gene_weight_matrix.index = snp_annotation_df.index
    snp_gene_weight_matrix.columns = snp_annotation_df.columns

    return snp_gene_weight_matrix


def calculate_ldscore_from_multiple_annotation(
    snp_annotation_df_list: list[pd.DataFrame],
    chrom: int,
    bfile_root: str,
    ld_wind: float = 1,
    ld_unit: str = "CM",
) -> list[pd.DataFrame]:
    """
    Calculate LD scores from multiple SNP annotation DataFrames.

    Parameters
    ----------
    snp_annotation_df_list : list
        List of DataFrames with SNP annotations
    chrom : int
        Chromosome number
    bfile_root : str
        Root path for PLINK bfiles
    ld_wind : float, optional
        LD window size, by default 1
    ld_unit : str, optional
        Unit for the LD window, by default "CM"

    Returns
    -------
    list
        List of DataFrames with calculated LD scores
    """
    # Combine annotations
    combined_annotations = pd.concat(snp_annotation_df_list, axis=1).astype(np.float32, copy=False)

    # Calculate LD scores
    combined_ld_scores = get_ldscore(
        bfile_root, chrom, combined_annotations.values, ld_wind=ld_wind, ld_unit=ld_unit
    )

    # Apply proper indices and columns
    combined_ld_scores.index = combined_annotations.index
    combined_ld_scores.columns = combined_annotations.columns

    # Split back into separate DataFrames
    annotation_lengths = [len(df.columns) for df in snp_annotation_df_list]
    result_dataframes = []
    start_col = 0

    for length in annotation_lengths:
        end_col = start_col + length
        result_dataframes.append(combined_ld_scores.iloc[:, start_col:end_col])
        start_col = end_col

    return result_dataframes


class LDScoreCalculator:
    """
    Class for calculating LD scores from gene specificity scores.

    This class handles the assignment of gene specificity scores to SNPs
    and the calculation of LD scores.
    """

    def __init__(self, config: GenerateLDScoreConfig):
        """
        Initialize LDScoreCalculator.

        Parameters
        ----------
        config : GenerateLDScoreConfig
            Configuration object
        """
        self.config = config
        self.validate_config()

        # Load marker scores
        self.mk_score = load_marker_score(config.mkscore_feather_path)

        # Load GTF and get common markers
        self.gtf_pr, self.mk_score_common = load_gtf(
            config.gtf_annotation_file, self.mk_score, window_size=config.gene_window_size
        )

        # Initialize enhancer data if provided
        self.enhancer_pr = self._initialize_enhancer() if config.enhancer_annotation_file else None

        # Initialize zarr file if needed
        self._initialize_zarr_if_needed()

    def validate_config(self):
        """Validate configuration parameters."""
        if not Path(self.config.mkscore_feather_path).exists():
            raise FileNotFoundError(
                f"Marker score file not found: {self.config.mkscore_feather_path}"
            )

        if not Path(self.config.gtf_annotation_file).exists():
            raise FileNotFoundError(
                f"GTF annotation file not found: {self.config.gtf_annotation_file}"
            )

        if (
            self.config.enhancer_annotation_file
            and not Path(self.config.enhancer_annotation_file).exists()
        ):
            raise FileNotFoundError(
                f"Enhancer annotation file not found: {self.config.enhancer_annotation_file}"
            )

    def _initialize_enhancer(self) -> pr.PyRanges:
        """
        Initialize enhancer data.

        Returns
        -------
        pr.PyRanges
            PyRanges object with enhancer data
        """
        # Load enhancer data
        enhancer_df = pr.read_bed(self.config.enhancer_annotation_file, as_df=True)
        enhancer_df.set_index("Name", inplace=True)
        enhancer_df.index.name = "gene_name"

        # Keep common genes and add marker score information
        avg_mkscore = pd.DataFrame(self.mk_score_common.mean(axis=1), columns=["avg_mkscore"])
        enhancer_df = enhancer_df.join(
            avg_mkscore,
            how="inner",
            on="gene_name",
        )

        # Add TSS information
        enhancer_df["TSS"] = self.gtf_pr.df.set_index("gene_name").reindex(enhancer_df.index)[
            "TSS"
        ]

        # Convert to PyRanges
        return pr.PyRanges(enhancer_df.reset_index())

    def _initialize_zarr_if_needed(self):
        """Initialize zarr file if zarr format is specified."""
        if self.config.ldscore_save_format == "zarr":
            chrom_snp_length_dict = get_snp_counts(self.config)
            self.chrom_snp_start_point = chrom_snp_length_dict["chrom_snp_start_point"]

            zarr_path = (
                Path(self.config.ldscore_save_dir) / f"{self.config.sample_name}.ldscore.zarr"
            )

            if not zarr_path.exists():
                self.zarr_file = zarr.open(
                    zarr_path.as_posix(),
                    mode="a",
                    dtype=np.float16,
                    chunks=self.config.zarr_chunk_size,
                    shape=(chrom_snp_length_dict["total"], self.mk_score_common.shape[1]),
                )
                zarr_path.parent.mkdir(parents=True, exist_ok=True)

                # Save metadata
                self.zarr_file.attrs["spot_names"] = self.mk_score_common.columns.to_list()
                self.zarr_file.attrs["chrom_snp_start_point"] = self.chrom_snp_start_point

            else:
                self.zarr_file = zarr.open(zarr_path.as_posix(), mode="a")

    def process_chromosome(self, chrom: int):
        """
        Process a single chromosome to calculate LD scores.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        logger.info(f"Processing chromosome {chrom}")

        # Get SNPs passing MAF filter
        self.snp_pass_maf = get_snp_pass_maf(self.config.bfile_root, chrom, maf_min=0.05)

        # Get SNP-gene dummy pairs
        self.snp_gene_pair_dummy = self._get_snp_gene_dummy(chrom)

        # Apply SNP filter if provided
        self._apply_snp_filter(chrom)

        # Process additional baseline annotations if provided
        if self.config.additional_baseline_annotation:
            self._process_additional_baseline(chrom)
        else:
            # Calculate SNP-gene weight matrix
            self.snp_gene_weight_matrix = calculate_ldscore_from_annotation(
                self.snp_gene_pair_dummy,
                chrom,
                self.config.bfile_root,
                ld_wind=self.config.ld_wind,
                ld_unit=self.config.ld_unit,
            )

            # Apply SNP filter if needed
            if self.keep_snp_mask is not None:
                self.snp_gene_weight_matrix = self.snp_gene_weight_matrix[self.keep_snp_mask]

        # Generate w_ld file if keep_snp_root is provided
        if self.config.keep_snp_root:
            self._generate_w_ld(chrom)

        # Save pre-calculated SNP-gene weight matrix if requested
        self._save_snp_gene_weight_matrix_if_needed(chrom)

        # Convert to sparse matrix for memory efficiency
        self.snp_gene_weight_matrix = csr_matrix(self.snp_gene_weight_matrix)
        logger.info(f"SNP-gene weight matrix shape: {self.snp_gene_weight_matrix.shape}")

        # Calculate baseline LD scores
        logger.info(f"Calculating baseline LD scores for chr{chrom}")
        self._calculate_baseline_ldscores(chrom)

        # Calculate LD scores for annotation
        logger.info(f"Calculating annotation LD scores for chr{chrom}")
        self._calculate_annotation_ldscores(chrom)

        # Clear memory
        self._clear_memory()

    def _generate_w_ld(self, chrom: int):
        """
        Generate w_ld file for the chromosome using filtered SNPs.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        if not self.config.keep_snp_root:
            logger.info(
                f"Skipping w_ld generation for chr{chrom} as keep_snp_root is not provided"
            )
            return

        logger.info(f"Generating w_ld for chr{chrom}")

        # Get the indices of SNPs to keep based on the keep_snp_mask
        keep_snps_index = np.nonzero(self.keep_snp_mask)[0]

        # Create a simple unit annotation (all ones) for the filtered SNPs
        unit_annotation = np.ones((len(keep_snps_index), 1))

        # Calculate LD scores using the filtered SNPs
        w_ld_scores = get_ldscore(
            self.config.bfile_root,
            chrom,
            unit_annotation,
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
            keep_snps_index=keep_snps_index.tolist(),
        )

        # Load the BIM file to get SNP information
        bim_data = pd.read_csv(
            f"{self.config.bfile_root}.{chrom}.bim",
            sep="\t",
            header=None,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        )

        # Get SNP names for the kept indices
        kept_snp_names = bim_data.iloc[keep_snps_index].SNP.tolist()

        # Create the w_ld DataFrame
        w_ld_df = pd.DataFrame(
            {
                "SNP": kept_snp_names,
                "L2": w_ld_scores.values.flatten(),
                "CHR": bim_data.iloc[keep_snps_index].CHR.values,
                "BP": bim_data.iloc[keep_snps_index].BP.values,
                "CM": bim_data.iloc[keep_snps_index].CM.values,
            }
        )

        # Reorder columns
        w_ld_df = w_ld_df[["CHR", "SNP", "BP", "CM", "L2"]]

        # Save to feather format
        w_ld_dir = Path(self.config.ldscore_save_dir) / "w_ld"
        w_ld_dir.mkdir(parents=True, exist_ok=True)
        w_ld_file = w_ld_dir / f"weights.{chrom}.l2.ldscore.gz"
        w_ld_df.to_csv(w_ld_file, sep="\t", index=False, compression="gzip")

        logger.info(f"Saved w_ld for chr{chrom} to {w_ld_file}")

    def _apply_snp_filter(self, chrom: int):
        """
        Apply SNP filter based on keep_snp_root.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        if self.config.keep_snp_root is not None:
            keep_snp_file = f"{self.config.keep_snp_root}.{chrom}.snp"
            keep_snp = pd.read_csv(keep_snp_file, header=None)[0].to_list()
            self.keep_snp_mask = self.snp_gene_pair_dummy.index.isin(keep_snp)
            self.snp_name = self.snp_gene_pair_dummy.index[self.keep_snp_mask].to_list()
            logger.info(f"Kept {len(self.snp_name)} SNPs after filtering with {keep_snp_file}")
            logger.info("These filtered SNPs will be used to calculate w_ld")
        else:
            self.keep_snp_mask = None
            self.snp_name = self.snp_gene_pair_dummy.index.to_list()
            logger.info(f"Using all {len(self.snp_name)} SNPs (no filter applied)")
            logger.warning("No keep_snp_root provided, all SNPs will be used to calculate w_ld.")

    def _process_additional_baseline(self, chrom: int):
        """
        Process additional baseline annotations.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        # Load additional baseline annotations
        additional_baseline_path = Path(self.config.additional_baseline_annotation)
        annot_file_path = additional_baseline_path / f"baseline.{chrom}.annot.gz"

        # Verify file existence
        if not annot_file_path.exists():
            raise FileNotFoundError(
                f"Additional baseline annotation file not found: {annot_file_path}"
            )

        # Load annotations
        additional_baseline_df = pd.read_csv(annot_file_path, sep="\t")
        additional_baseline_df.set_index("SNP", inplace=True)

        # Drop unnecessary columns
        for col in ["CHR", "BP", "CM"]:
            if col in additional_baseline_df.columns:
                additional_baseline_df.drop(col, axis=1, inplace=True)

        # Check for SNPs not in the additional baseline
        missing_snps = ~self.snp_gene_pair_dummy.index.isin(additional_baseline_df.index)
        missing_count = missing_snps.sum()

        if missing_count > 0:
            logger.warning(
                f"{missing_count} SNPs not found in additional baseline annotations. "
                "Setting their values to 0."
            )
            additional_baseline_df = additional_baseline_df.reindex(
                self.snp_gene_pair_dummy.index, fill_value=0
            )
        else:
            additional_baseline_df = additional_baseline_df.reindex(self.snp_gene_pair_dummy.index)

        # Calculate LD scores for both annotation sets together
        self.snp_gene_weight_matrix, additional_ldscore = (
            calculate_ldscore_from_multiple_annotation(
                [self.snp_gene_pair_dummy, additional_baseline_df],
                chrom,
                self.config.bfile_root,
                ld_wind=self.config.ld_wind,
                ld_unit=self.config.ld_unit,
            )
        )

        # Filter additional ldscore
        additional_ldscore = additional_ldscore.loc[self.snp_name]

        # Save additional baseline LD scores
        ld_score_file = f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.ldscore.feather"
        m_file_path = f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.M"
        m_5_file_path = (
            f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.M_5_50"
        )
        Path(m_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save LD scores
        self._save_ldscore_to_feather(
            additional_ldscore.values,
            column_names=additional_ldscore.columns,
            save_file_name=ld_score_file,
        )

        # Calculate and save M values
        m_chr_chunk = additional_baseline_df.values.sum(axis=0, keepdims=True)
        m_5_chr_chunk = additional_baseline_df.loc[self.snp_pass_maf].values.sum(
            axis=0, keepdims=True
        )

        # Save M statistics
        np.savetxt(m_file_path, m_chr_chunk, delimiter="\t")
        np.savetxt(m_5_file_path, m_5_chr_chunk, delimiter="\t")

    def _save_snp_gene_weight_matrix_if_needed(self, chrom: int):
        """
        Save pre-calculated SNP-gene weight matrix if requested.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        if self.config.save_pre_calculate_snp_gene_weight_matrix:
            save_dir = Path(self.config.ldscore_save_dir) / "snp_gene_weight_matrix"
            save_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving SNP-gene weight matrix for chr{chrom}")

            save_path = save_dir / f"{chrom}.snp_gene_weight_matrix.feather"
            self.snp_gene_weight_matrix.reset_index().to_feather(save_path)

    def _calculate_baseline_ldscores(self, chrom: int):
        """
        Calculate and save baseline LD scores.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        # Create baseline scores
        baseline_mk_score = np.ones((self.snp_gene_pair_dummy.shape[1], 2))
        baseline_mk_score[-1, 0] = 0  # all_gene column

        baseline_df = pd.DataFrame(
            baseline_mk_score, index=self.snp_gene_pair_dummy.columns, columns=["all_gene", "base"]
        )

        # Define file paths
        ld_score_file = (
            f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.ldscore.feather"
        )
        m_file = f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.M"
        m_5_file = f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.M_5_50"

        # Calculate LD scores
        ldscore_chunk = self._calculate_ldscore_from_weights(baseline_df, drop_dummy_na=False)

        # Save LD scores and M values
        self._save_ldscore_to_feather(
            ldscore_chunk,
            column_names=baseline_df.columns,
            save_file_name=ld_score_file,
        )

        self._calculate_and_save_m_values(
            baseline_df,
            m_file,
            m_5_file,
            drop_dummy_na=False,
        )

        # If keep_snp_root is not provided, use the first column of baseline ldscore as w_ld
        if not self.config.keep_snp_root:
            self._save_baseline_as_w_ld(chrom, ldscore_chunk)

    def _save_baseline_as_w_ld(self, chrom: int, ldscore_chunk: np.ndarray):
        """
        Save the first column of baseline ldscore as w_ld.

        Parameters
        ----------
        chrom : int
            Chromosome number
        ldscore_chunk : np.ndarray
            Array with baseline LD scores
        """
        logger.info(f"Using first column of baseline ldscore as w_ld for chr{chrom}")

        # Create w_ld directory
        w_ld_dir = Path(self.config.ldscore_save_dir) / "w_ld"
        w_ld_dir.mkdir(parents=True, exist_ok=True)

        # Define file path
        w_ld_file = w_ld_dir / f"weights.{chrom}.l2.ldscore.gz"

        # Extract the first column
        w_ld_values = ldscore_chunk[:, 0]

        # Create a DataFrame
        bim_data = pd.read_csv(
            f"{self.config.bfile_root}.{chrom}.bim",
            sep="\t",
            header=None,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        )
        w_ld_df = pd.DataFrame(
            {
                "SNP": self.snp_name,
                "L2": w_ld_values,
            }
        )

        # Add CHR, BP, and CM information
        w_ld_df = w_ld_df.merge(bim_data[["SNP", "CHR", "BP", "CM"]], on="SNP", how="left")

        # Reorder columns
        w_ld_df = w_ld_df[["CHR", "SNP", "BP", "CM", "L2"]]

        w_ld_df.to_csv(w_ld_file, sep="\t", index=False, compression="gzip")

        logger.info(f"Saved w_ld for chr{chrom} to {w_ld_file}")

    def _calculate_annotation_ldscores(self, chrom: int):
        """
        Calculate and save LD scores for spatial annotations.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        # Get marker scores for gene columns (excluding dummy NA column)
        mk_scores = self.mk_score_common.loc[self.snp_gene_pair_dummy.columns[:-1]]

        # Process in chunks
        chunk_index = 1
        for i in trange(
            0,
            mk_scores.shape[1],
            self.config.spots_per_chunk,
            desc=f"Calculating LD scores for chr{chrom}",
        ):
            # Get marker scores for current chunk
            mk_score_chunk = mk_scores.iloc[:, i : i + self.config.spots_per_chunk]

            # Define file paths
            sample_name = self.config.sample_name
            ld_score_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.ldscore.feather"
            m_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.M"
            m_5_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.M_5_50"

            # Calculate LD scores
            ldscore_chunk = self._calculate_ldscore_from_weights(mk_score_chunk)

            # Save LD scores based on format
            if self.config.ldscore_save_format == "feather":
                self._save_ldscore_to_feather(
                    ldscore_chunk,
                    column_names=mk_score_chunk.columns,
                    save_file_name=ld_score_file,
                )
            elif self.config.ldscore_save_format == "zarr":
                self._save_ldscore_chunk_to_zarr(
                    ldscore_chunk,
                    chrom=chrom,
                    start_col_index=i,
                )
            else:
                raise ValueError(f"Invalid ldscore_save_format: {self.config.ldscore_save_format}")

            # Calculate and save M values
            self._calculate_and_save_m_values(
                mk_score_chunk,
                m_file,
                m_5_file,
                drop_dummy_na=True,
            )

            chunk_index += 1

            # Clear memory
            del ldscore_chunk
            gc.collect()

    def _calculate_ldscore_from_weights(
        self, marker_scores: pd.DataFrame, drop_dummy_na: bool = True
    ) -> np.ndarray:
        """
        Calculate LD scores using SNP-gene weight matrix.

        Parameters
        ----------
        marker_scores : pd.DataFrame
            DataFrame with marker scores
        drop_dummy_na : bool, optional
            Whether to drop the dummy NA column, by default True

        Returns
        -------
        np.ndarray
            Array with calculated LD scores
        """
        weight_matrix = self.snp_gene_weight_matrix

        if drop_dummy_na:
            # Use all columns except the last one (dummy NA)
            ldscore = weight_matrix[:, :-1] @ marker_scores
        else:
            ldscore = weight_matrix @ marker_scores

        return ldscore

    def _save_ldscore_to_feather(
        self, ldscore_data: np.ndarray, column_names: list[str], save_file_name: str
    ):
        """
        Save LD scores to a feather file.

        Parameters
        ----------
        ldscore_data : np.ndarray
            Array with LD scores
        column_names : list
            List of column names
        save_file_name : str
            Path to save the feather file
        """
        # Create directory if needed
        save_dir = Path(save_file_name).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert to float16 for storage efficiency
        ldscore_data = ldscore_data.astype(np.float16, copy=False)

        # Handle numerical overflow
        ldscore_data[np.isinf(ldscore_data)] = np.finfo(np.float16).max

        # Create DataFrame and save
        df = pd.DataFrame(
            ldscore_data,
            index=self.snp_name,
            columns=column_names,
        )
        df.index.name = "SNP"
        df.reset_index().to_feather(save_file_name)

    def _save_ldscore_chunk_to_zarr(
        self, ldscore_data: np.ndarray, chrom: int, start_col_index: int
    ):
        """
        Save LD scores to a zarr array.

        Parameters
        ----------
        ldscore_data : np.ndarray
            Array with LD scores
        chrom : int
            Chromosome number
        start_col_index : int
            Starting column index in the zarr array
        """
        # Convert to float16 for storage efficiency
        ldscore_data = ldscore_data.astype(np.float16, copy=False)

        # Handle numerical overflow
        ldscore_data[np.isinf(ldscore_data)] = np.finfo(np.float16).max

        # Get start and end indices for this chromosome
        chrom_start = self.chrom_snp_start_point[chrom - 1]
        chrom_end = self.chrom_snp_start_point[chrom]

        # Save to zarr array
        self.zarr_file[
            chrom_start:chrom_end,
            start_col_index : start_col_index + ldscore_data.shape[1],
        ] = ldscore_data

    def _calculate_and_save_m_values(
        self,
        marker_scores: pd.DataFrame,
        m_file_path: str,
        m_5_file_path: str,
        drop_dummy_na: bool = True,
    ):
        """
        Calculate and save M statistics.

        Parameters
        ----------
        marker_scores : pd.DataFrame
            DataFrame with marker scores
        m_file_path : str
            Path to save M values
        m_5_file_path : str
            Path to save M_5_50 values
        drop_dummy_na : bool, optional
            Whether to drop the dummy NA column, by default True
        """
        # Create directory if needed
        save_dir = Path(m_file_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get sum of SNP-gene pairs
        snp_gene_sum = self.snp_gene_pair_dummy.values.sum(axis=0, keepdims=True)
        snp_gene_sum_maf = self.snp_gene_pair_dummy.loc[self.snp_pass_maf].values.sum(
            axis=0, keepdims=True
        )

        # Drop dummy NA column if requested
        if drop_dummy_na:
            snp_gene_sum = snp_gene_sum[:, :-1]
            snp_gene_sum_maf = snp_gene_sum_maf[:, :-1]

        # Calculate M values
        m_values = snp_gene_sum @ marker_scores
        m_5_values = snp_gene_sum_maf @ marker_scores

        # Save M values
        np.savetxt(m_file_path, m_values, delimiter="\t")
        np.savetxt(m_5_file_path, m_5_values, delimiter="\t")

    def _get_snp_gene_dummy(self, chrom: int) -> pd.DataFrame:
        """
        Get dummy matrix for SNP-gene pairs.

        Parameters
        ----------
        chrom : int
            Chromosome number

        Returns
        -------
        pd.DataFrame
            DataFrame with dummy variables for SNP-gene pairs
        """
        logger.info(f"Creating SNP-gene mappings for chromosome {chrom}")

        # Load BIM file
        bim, bim_pr = load_bim(self.config.bfile_root, chrom)

        # Determine mapping strategy
        if self.config.gene_window_enhancer_priority in ["gene_window_first", "enhancer_first"]:
            # Use both gene window and enhancer
            snp_gene_pair = self._combine_gtf_and_enhancer_mappings(bim, bim_pr)

        elif self.config.gene_window_enhancer_priority is None:
            # Use only gene window
            snp_gene_pair = self._get_snp_gene_pair_from_gtf(bim, bim_pr)

        elif self.config.gene_window_enhancer_priority == "enhancer_only":
            # Use only enhancer
            snp_gene_pair = self._get_snp_gene_pair_from_enhancer(bim, bim_pr)

        else:
            raise ValueError(
                f"Invalid gene_window_enhancer_priority: {self.config.gene_window_enhancer_priority}"
            )

        # Save SNP-gene pair mapping
        self._save_snp_gene_pair_mapping(snp_gene_pair, chrom)

        # Create dummy variables
        snp_gene_dummy = pd.get_dummies(snp_gene_pair["gene_name"], dummy_na=True)

        return snp_gene_dummy

    def _combine_gtf_and_enhancer_mappings(
        self, bim: pd.DataFrame, bim_pr: pr.PyRanges
    ) -> pd.DataFrame:
        """
        Combine gene window and enhancer mappings.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            Combined SNP-gene pair mapping
        """
        # Get mappings from both sources
        gtf_mapping = self._get_snp_gene_pair_from_gtf(bim, bim_pr)
        enhancer_mapping = self._get_snp_gene_pair_from_enhancer(bim, bim_pr)

        # Find SNPs with missing mappings in each source
        mask_of_nan_gtf = gtf_mapping.gene_name.isna()
        mask_of_nan_enhancer = enhancer_mapping.gene_name.isna()

        # Combine based on priority
        if self.config.gene_window_enhancer_priority == "gene_window_first":
            # Use gene window mappings first, fill missing with enhancer mappings
            combined_mapping = gtf_mapping.copy()
            combined_mapping.loc[mask_of_nan_gtf, "gene_name"] = enhancer_mapping.loc[
                mask_of_nan_gtf, "gene_name"
            ]
            logger.info(
                f"Filled {mask_of_nan_gtf.sum()} SNPs with no GTF mapping using enhancer mappings"
            )

        elif self.config.gene_window_enhancer_priority == "enhancer_first":
            # Use enhancer mappings first, fill missing with gene window mappings
            combined_mapping = enhancer_mapping.copy()
            combined_mapping.loc[mask_of_nan_enhancer, "gene_name"] = gtf_mapping.loc[
                mask_of_nan_enhancer, "gene_name"
            ]
            logger.info(
                f"Filled {mask_of_nan_enhancer.sum()} SNPs with no enhancer mapping using GTF mappings"
            )

        else:
            raise ValueError(
                f"Invalid gene_window_enhancer_priority for combining: {self.config.gene_window_enhancer_priority}"
            )

        return combined_mapping

    def _get_snp_gene_pair_from_gtf(self, bim: pd.DataFrame, bim_pr: pr.PyRanges) -> pd.DataFrame:
        """
        Get SNP-gene pairs based on GTF annotations.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            SNP-gene pairs based on GTF
        """
        logger.info(
            "Getting SNP-gene pairs from GTF. SNPs in multiple genes will be assigned to the nearest gene (by TSS)"
        )

        # Find overlaps between SNPs and gene windows
        overlaps = overlaps_gtf_bim(self.gtf_pr, bim_pr)

        # Get SNP information
        annot = bim[["CHR", "BP", "SNP", "CM"]]

        # Create SNP-gene pairs DataFrame
        snp_gene_pair = (
            overlaps[["SNP", "gene_name"]]
            .set_index("SNP")
            .join(annot.set_index("SNP"), how="right")
        )

        logger.info(f"Found {overlaps.shape[0]} SNP-gene pairs from GTF")

        return snp_gene_pair

    def _get_snp_gene_pair_from_enhancer(
        self, bim: pd.DataFrame, bim_pr: pr.PyRanges
    ) -> pd.DataFrame:
        """
        Get SNP-gene pairs based on enhancer annotations.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            SNP-gene pairs based on enhancer
        """
        if self.enhancer_pr is None:
            raise ValueError("Enhancer annotation file is required but not provided")

        # Find overlaps between SNPs and enhancers
        overlaps = self.enhancer_pr.join(bim_pr).df

        # Get SNP information
        annot = bim[["CHR", "BP", "SNP", "CM"]]

        if self.config.snp_multiple_enhancer_strategy == "max_mkscore":
            logger.info(
                "SNPs in multiple enhancers will be assigned to the gene with highest marker score"
            )
            overlaps = overlaps.loc[overlaps.groupby("SNP").avg_mkscore.idxmax()]

        elif self.config.snp_multiple_enhancer_strategy == "nearest_TSS":
            logger.info("SNPs in multiple enhancers will be assigned to the gene with nearest TSS")
            overlaps["Distance"] = np.abs(overlaps["Start_b"] - overlaps["TSS"])
            overlaps = overlaps.loc[overlaps.groupby("SNP").Distance.idxmin()]

        # Create SNP-gene pairs DataFrame
        snp_gene_pair = (
            overlaps[["SNP", "gene_name"]]
            .set_index("SNP")
            .join(annot.set_index("SNP"), how="right")
        )

        logger.info(f"Found {overlaps.shape[0]} SNP-gene pairs from enhancers")

        return snp_gene_pair

    def _save_snp_gene_pair_mapping(self, snp_gene_pair: pd.DataFrame, chrom: int):
        """
        Save SNP-gene pair mapping to a feather file.

        Parameters
        ----------
        snp_gene_pair : pd.DataFrame
            SNP-gene pair mapping
        chrom : int
            Chromosome number
        """
        save_path = (
            Path(self.config.ldscore_save_dir) / f"SNP_gene_pair/SNP_gene_pair_chr{chrom}.feather"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        snp_gene_pair.reset_index().to_feather(save_path)

    def _clear_memory(self):
        """Clear memory to prevent leaks."""
        gc.collect()


def run_generate_ldscore(config: GenerateLDScoreConfig):
    """
    Main function to run the LD score generation.

    Parameters
    ----------
    config : GenerateLDScoreConfig
        Configuration object
    """
    # Create output directory
    Path(config.ldscore_save_dir).mkdir(parents=True, exist_ok=True)

    if config.ldscore_save_format == "quick_mode":
        logger.info(
            "Running in quick_mode. Skip the process of generating ldscore. Using the pre-calculated ldscore."
        )
        ldscore_save_dir = Path(config.ldscore_save_dir)

        # Set up symbolic links
        baseline_dir = ldscore_save_dir / "baseline"
        baseline_dir.parent.mkdir(parents=True, exist_ok=True)
        if not baseline_dir.exists():
            baseline_dir.symlink_to(config.baseline_annotation_dir, target_is_directory=True)

        snp_gene_pair_dir = ldscore_save_dir / "SNP_gene_pair"
        snp_gene_pair_dir.parent.mkdir(parents=True, exist_ok=True)
        if not snp_gene_pair_dir.exists():
            snp_gene_pair_dir.symlink_to(config.SNP_gene_pair_dir, target_is_directory=True)

        # Create a done file to mark completion
        done_file = ldscore_save_dir / f"{config.sample_name}_generate_ldscore.done"
        done_file.touch()

        return

    # Initialize calculator
    calculator = LDScoreCalculator(config)

    # Process chromosomes
    if config.chrom == "all":
        # Process all chromosomes
        for chrom in range(1, 23):
            try:
                calculator.process_chromosome(chrom)
            except Exception as e:
                logger.error(f"Error processing chromosome {chrom}: {e}")
                raise
    else:
        # Process one chromosome
        try:
            chrom = int(config.chrom)
        except ValueError:
            logger.error(f"Invalid chromosome: {config.chrom}")
            raise ValueError(
                f"Invalid chromosome: {config.chrom}. Must be an integer between 1-22 or 'all'"
            ) from None
        else:
            calculator.process_chromosome(chrom)

    # Create a done file to mark completion
    done_file = Path(config.ldscore_save_dir) / f"{config.sample_name}_generate_ldscore.done"
    done_file.touch()

    logger.info(f"LD score generation completed for {config.sample_name}")
