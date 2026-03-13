from mcp.server.fastmcp import FastMCP
from .tools import *
from .docs import search_docs

mcp = FastMCP("gsmap")


@mcp.tool()
def check_gsmap_installation():
    """
    Check whether the gsMap command-line tool is available.

    This tool verifies that gsMap is correctly installed
    and accessible in the system PATH.

    Use this tool when debugging installation problems
    or before running any gsMap pipeline.
    """
    return check_gsmap()


@mcp.tool()
def download_gsmap_resource_data(output_dir: str):
    """
    Download the gsMap reference resource dataset.

    The reference resource includes genome annotations,
    LD reference data, and other resources required
    for running gsMap spatial enrichment analysis.

    Parameters
    ----------
    output_dir : str
        Directory where the resource dataset will be downloaded
        and extracted.

    This step is required before running gsMap pipelines
    if the reference resource has not already been installed.
    """
    return download_gsmap_resource(output_dir)


@mcp.tool()
def download_example_dataset(output_dir: str):
    """
    Download the official gsMap example spatial transcriptomics dataset.

    This dataset can be used to test the gsMap pipeline
    and reproduce the demo analysis.

    Parameters
    ----------
    output_dir : str
        Directory where the example dataset will be downloaded
        and extracted.
    """
    return download_example_data(output_dir)


@mcp.tool()
def gsmap_format_sumstats(
    sumstats: str,
    out: str,
    args: list[str] = []
):
    """
    Convert GWAS summary statistics into gsMap-compatible format.

    Many GWAS datasets use different column naming conventions.
    This tool converts them into the standard gsMap format.

    Required columns in the final gsMap format:

    SNP
    A1
    A2
    Z
    N

    Parameters
    ----------
    sumstats : str
        Path to the original GWAS summary statistics file.

    out : str
        Path where the formatted GWAS file will be written.

    args : list[str]
        Optional additional CLI flags passed directly to
        `gsmap format_sumstats`.

        Example:

        args=[
            "--snp","rsid",
            "--a1","EA",
            "--a2","NEA",
            "--p","P",
            "--n","N"
        ]

    Use this tool when GWAS column names differ from
    the gsMap default format.
    """
    return format_sumstats(sumstats, out, args)


@mcp.tool()
def gsmap_quick_mode(
    workdir: str,
    hdf5_path: str,
    sumstats_file: str,
    trait_name: str,
    args: list[str] = []
):
    """
    Run the gsMap quick_mode pipeline.

    This is the recommended default workflow for most gsMap analyses.

    The quick_mode pipeline automatically performs:

    1. Latent spatial representation learning
    2. Gene spatial score calculation
    3. LD score generation
    4. Spatial LDSC enrichment analysis

    Parameters
    ----------
    workdir : str
        Output directory where all results will be saved.

    hdf5_path : str
        Path to the spatial transcriptomics dataset (.h5ad).

    sumstats_file : str
        Path to the GWAS summary statistics file.

    trait_name : str
        Name of the GWAS trait being analyzed.

    args : list[str]
        Optional additional gsMap CLI flags.

        Example:

        args=[
            "--annotation","celltype",
            "--data_layer","counts"
        ]

    Use this tool for most spatial GWAS analyses.
    """
    return run_quick_mode(
        workdir,
        hdf5_path,
        sumstats_file,
        trait_name,
        args
    )


@mcp.tool()
def gsmap_find_latent_representation(
    workdir: str,
    sample_name: str,
    hdf5_path: str,
    args: list[str] = []
):
    """
    Step 1 of the gsMap step-by-step pipeline.

    Learn latent spatial representations from the spatial
    transcriptomics dataset.

    Parameters
    ----------
    workdir : str
        Directory where intermediate outputs will be saved.

    sample_name : str
        Name of the spatial transcriptomics sample.

    hdf5_path : str
        Path to the spatial transcriptomics dataset (.h5ad).

    args : list[str]
        Optional additional CLI parameters for the latent
        representation learning step.
    """
    return find_latent_representation(
        workdir,
        sample_name,
        hdf5_path,
        args
    )


@mcp.tool()
def gsmap_latent_to_gene(
    workdir: str,
    sample_name: str,
    hdf5_path: str,
    args: list[str] = []
):
    """
    Step 2 of the gsMap step-by-step pipeline.

    Convert latent spatial representations into gene-level
    spatial scores.

    Parameters
    ----------
    workdir : str
        Directory containing previous pipeline outputs.

    sample_name : str
        Name of the spatial transcriptomics sample.

    hdf5_path : str
        Path to the spatial transcriptomics dataset.

    args : list[str]
        Optional additional CLI flags.
    """
    return latent_to_gene(
        workdir,
        sample_name,
        hdf5_path,
        args
    )


@mcp.tool()
def gsmap_generate_ldscore(
    workdir: str,
    sample_name: str,
    chromosome: int,
    args: list[str] = []
):
    """
    Step 3 of the gsMap step-by-step pipeline.

    Generate LD scores for each spatial gene annotation.

    Parameters
    ----------
    workdir : str
        Directory containing intermediate pipeline outputs.

    sample_name : str
        Name of the spatial transcriptomics sample.

    chromosome : int
        Chromosome number for LD score generation.

    args : list[str]
        Optional additional CLI flags.
    """
    return generate_ldscore(
        workdir,
        sample_name,
        chromosome,
        args
    )


@mcp.tool()
def gsmap_spatial_ldsc(
    workdir: str,
    sample_name: str,
    sumstats_file: str,
    trait_name: str,
    args: list[str] = []
):
    """
    Step 4 of the gsMap step-by-step pipeline.

    Run spatial LDSC regression to identify spatial regions
    enriched for GWAS signals.

    Parameters
    ----------
    workdir : str
        Directory containing LD scores and intermediate outputs.

    sample_name : str
        Name of the spatial transcriptomics sample.

    sumstats_file : str
        GWAS summary statistics file.

    trait_name : str
        Name of the GWAS trait.

    args : list[str]
        Optional additional CLI parameters.
    """
    return spatial_ldsc(
        workdir,
        sample_name,
        sumstats_file,
        trait_name,
        args
    )


@mcp.tool()
def search_gsmap_docs(query: str):
    """
    Search the official gsMap documentation.

    Use this tool when the meaning of a CLI parameter
    or command option is unclear.

    Example queries:

    format_sumstats flags
    quick_mode parameters
    latent representation options
    data format requirements
    """
    return search_docs(query)


def main():
    mcp.run()