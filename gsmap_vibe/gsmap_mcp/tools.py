import subprocess
import urllib.request
import tarfile
from pathlib import Path


GSMAP_RESOURCE_URL = "https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz"
GSMAP_EXAMPLE_URL = "https://yanglab.westlake.edu.cn/data/gsMap/Visium_example_data.tar.gz"


def run_command(cmd):

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def check_gsmap():

    try:
        run_command(["gsmap", "--help"])
        return "gsMap CLI detected"
    except:
        return "gsMap CLI not found"


def download_gsmap_resource(output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "gsMap_resource.tar.gz"

    urllib.request.urlretrieve(GSMAP_RESOURCE_URL, tar_path)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)

    return f"gsMap resource downloaded to {output_dir}"


def download_example_data(output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "Visium_example_data.tar.gz"

    urllib.request.urlretrieve(GSMAP_EXAMPLE_URL, tar_path)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)

    return f"Example dataset downloaded to {output_dir}"


def format_sumstats(sumstats, out, args):

    cmd = [
        "gsmap",
        "format_sumstats",
        "--sumstats", sumstats,
        "--out", out,
    ]

    cmd.extend(args)

    run_command(cmd)

    return f"Formatted GWAS saved to {out}"


def run_quick_mode(workdir, hdf5_path, sumstats_file, trait_name, args):

    cmd = [
        "gsmap",
        "quick_mode",
        "--workdir", workdir,
        "--hdf5_path", hdf5_path,
        "--sumstats_file", sumstats_file,
        "--trait_name", trait_name,
    ]

    cmd.extend(args)

    run_command(cmd)

    return "gsMap quick_mode finished"


def find_latent_representation(workdir, sample_name, hdf5_path, args):

    cmd = [
        "gsmap",
        "run_find_latent_representations",
        "--workdir", workdir,
        "--sample_name", sample_name,
        "--input_hdf5_path", hdf5_path,
    ]

    cmd.extend(args)

    run_command(cmd)

    return "Latent representation generated"


def latent_to_gene(workdir, sample_name, hdf5_path, args):

    cmd = [
        "gsmap",
        "run_latent_to_gene",
        "--workdir", workdir,
        "--sample_name", sample_name,
        "--input_hdf5_path", hdf5_path,
    ]

    cmd.extend(args)

    run_command(cmd)

    return "Gene spatial scores generated"


def generate_ldscore(workdir, sample_name, chromosome, args):

    cmd = [
        "gsmap",
        "run_generate_ldscore",
        "--workdir", workdir,
        "--sample_name", sample_name,
        "--chrom", str(chromosome),
    ]

    cmd.extend(args)

    run_command(cmd)

    return f"LD score generated for chromosome {chromosome}"


def spatial_ldsc(workdir, sample_name, sumstats_file, trait_name, args):

    cmd = [
        "gsmap",
        "run_spatial_ldsc",
        "--workdir", workdir,
        "--sample_name", sample_name,
        "--sumstats_file", sumstats_file,
        "--trait_name", trait_name,
    ]

    cmd.extend(args)

    run_command(cmd)

    return "Spatial LDSC finished"