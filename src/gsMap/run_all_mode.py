import logging
import time
from pathlib import Path

from gsMap.cauchy_combination_test import run_Cauchy_combination
from gsMap.config import GenerateLDScoreConfig, SpatialLDSCConfig, LatentToGeneConfig, \
    FindLatentRepresentationsConfig, CauchyCombinationConfig, RunAllModeConfig, ReportConfig
from gsMap.find_latent_representation import run_find_latent_representation
from gsMap.generate_ldscore import run_generate_ldscore
from gsMap.latent_to_gene import run_latent_to_gene
from gsMap.report import run_report
from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc



def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def run_pipeline(config: RunAllModeConfig):
    # # Set up logging
    log_file = Path(config.workdir) / config.sample_name / 'gsMap_pipeline.log'
    logging.basicConfig(level=logging.INFO,
                        format='[{asctime}] {levelname:8s} {name} {message}',
                        handlers=[
                            logging.FileHandler(log_file),
                        ],
                        style='{')

    logger = logging.getLogger('gsMap Pipeline')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[{asctime}] {name} {levelname:8s} {message}', style='{'))
    logger.addHandler(handler)

    logger.info("Starting pipeline with configuration: %s", config)

    find_latent_config = FindLatentRepresentationsConfig(
        workdir=config.workdir,
        input_hdf5_path=config.hdf5_path,
        sample_name=config.sample_name,
        annotation=config.annotation,
        data_layer=config.data_layer
    )

    latent_to_gene_config = LatentToGeneConfig(
        workdir=config.workdir,
        sample_name=config.sample_name,
        annotation=config.annotation,
        latent_representation='latent_GVAE',
        num_neighbour=51,
        num_neighbour_spatial=201,
        homolog_file=config.homolog_file
    )

    ldscore_config = GenerateLDScoreConfig(
        workdir=config.workdir,
        sample_name=config.sample_name,
        chrom='all',
        # ldscore_save_dir=f"{config.workdir}/{config.sample_name}/generate_ldscore",
        # mkscore_feather_file=latent_to_gene_config.output_feather_path,
        bfile_root=config.bfile_root,
        keep_snp_root=config.keep_snp_root,
        gtf_annotation_file=config.gtffile,
        spots_per_chunk=5_000,
        baseline_annotation_dir=config.baseline_annotation_dir,
        SNP_gene_pair_dir=config.SNP_gene_pair_dir,
        ldscore_save_format='quick_mode'

    )

    pipeline_start_time = time.time()

    # Step 1: Find latent representations
    start_time = time.time()
    logger.info("Step 1: Finding latent representations")
    if Path(find_latent_config.hdf5_with_latent_path).exists():
        logger.info(
            f"Find latent representations already done. Results saved at {find_latent_config.hdf5_with_latent_path}. Skipping...")
    else:
        run_find_latent_representation(find_latent_config)
    end_time = time.time()
    logger.info(f"Step 1 completed in {format_duration(end_time - start_time)}.")

    # Step 2: Latent to gene
    start_time = time.time()
    logger.info("Step 2: Mapping latent representations to genes")
    if Path(latent_to_gene_config.mkscore_feather_path).exists():
        logger.info(
            f"Latent to gene mapping already done. Results saved at {latent_to_gene_config.mkscore_feather_path}. Skipping...")
    else:
        run_latent_to_gene(latent_to_gene_config)
    end_time = time.time()
    logger.info(f"Step 2 completed in {format_duration(end_time - start_time)}.")

    # Step 3: Generate LDScores
    start_time = time.time()
    logger.info("Step 3: Generating LDScores")

    # check if LDscore has been generated by the done file
    ldsc_done_file = Path(ldscore_config.ldscore_save_dir) / f"{config.sample_name}_generate_ldscore.done"
    if ldsc_done_file.exists():
        logger.info(f"LDScore generation already done. Results saved at {ldscore_config.ldscore_save_dir}. Skipping...")
    else:
        run_generate_ldscore(ldscore_config)
        end_time = time.time()
        logger.info(f"Step 3 completed in {format_duration(end_time - start_time)}.")
        # create a done file
        ldsc_done_file.touch()

    # Step 4: Spatial LDSC
    start_time = time.time()
    logger.info("Step 4: Running spatial LDSC")

    sumstats_config = config.sumstats_config_dict
    for trait_name in sumstats_config:
        logger.info("Running spatial LDSC for trait: %s", trait_name)
        # detect if the spatial LDSC has been done:
        spatial_ldsc_result_file = Path(config.ldsc_save_dir) / f"{config.sample_name}_{trait_name}.csv.gz"

        if spatial_ldsc_result_file.exists():
            logger.info(
                f"Spatial LDSC already done for trait {trait_name}. Results saved at {spatial_ldsc_result_file}. Skipping...")
            continue

        spatial_ldsc_config_trait = SpatialLDSCConfig(
            workdir=config.workdir,
            sumstats_file=sumstats_config[trait_name],
            trait_name=trait_name,
            w_file=config.w_file,
            sample_name=config.sample_name,
            # ldscore_save_dir=spatial_ldsc_config.ldscore_save_dir,
            # ldsc_save_dir=spatial_ldsc_config.ldsc_save_dir,
            num_processes=config.max_processes,
            ldscore_save_format='quick_mode',
            snp_gene_weight_adata_path=config.snp_gene_weight_adata_path,
        )
        run_spatial_ldsc(spatial_ldsc_config_trait)
    end_time = time.time()
    logger.info(f"Step 4 completed in {format_duration(end_time - start_time)}.")

    # Step 5: Cauchy combination test
    start_time = time.time()
    logger.info("Step 6: Running Cauchy combination test")
    '/storage/yangjianLab/chenwenhao/projects/202312_GPS/test/20240817_vanilla_pipeline_mouse_embryo_v4/E16.5_E1S1.MOSTA/cauchy_combination/E16.5_E1S1.MOSTA_Depression_2023_NatureMed.Cauchy.csv.gz'
    for trait_name in sumstats_config:
        # check if the cauchy combination has been done
        cauchy_result_file = config.get_cauchy_result_file(trait_name)
        if cauchy_result_file.exists():
            logger.info(
                f"Cauchy combination already done for trait {trait_name}. Results saved at {cauchy_result_file}. Skipping...")
            continue
        cauchy_config = CauchyCombinationConfig(
            workdir=config.workdir,
            sample_name=config.sample_name,
            annotation=config.annotation,
            trait_name=trait_name,
        )
        run_Cauchy_combination(cauchy_config)
    end_time = time.time()
    logger.info(f"Step 5 completed in {format_duration(end_time - start_time)}.")

    # Step 7:
    # start_time = time.time()
    # logger.info("Step 7: Running diagnosis")
    # for trait_name in sumstats_config:
    #     diagnosis_config = DiagnosisConfig(
    #         workdir=config.workdir,
    #         sample_name=config.sample_name,
    #         # input_hdf5_path=config.hdf5_path,
    #         annotation=config.annotation,
    #         # mkscore_feather_file=latent_to_gene_config.output_feather_path,
    #         # input_ldsc_dir=spatial_ldsc_config.ldsc_save_dir,
    #         trait_name=trait_name,
    #         plot_type='manhattan',
    #         # ldscore_save_dir=f"{config.workdir}/{config.sample_name}/generate_ldscore",
    #         top_corr_genes=50,
    #         selected_genes=None,
    #         sumstats_file=sumstats_config[trait_name],
    #         # diagnosis_save_dir=f"{config.workdir}/{config.sample_name}/diagnosis"
    #     )
    #     run_Diagnosis(diagnosis_config)
    #
    #     diagnosis_config.plot_type = 'GSS'
    #     diagnosis_config.top_corr_genes = 20
    #     run_Diagnosis(diagnosis_config)
    # end_time = time.time()
    # logger.info(f"Step 7 completed in {format_duration(end_time - start_time)}.")
    #

    # Step 8: Report

    for trait_name in sumstats_config:
        logger.info("Running final report generation for trait: %s", trait_name)
        report_config = ReportConfig(
            workdir=config.workdir,
            sample_name=config.sample_name,
            annotation=config.annotation,
            trait_name=trait_name,
            plot_type='all',
            top_corr_genes=50,
            selected_genes=None,
            sumstats_file=sumstats_config[trait_name],
        )
        # Create the run parameters dictionary for each trait
        run_parameter_dict = {
            "sample_name": config.sample_name,
            "trait_name": trait_name,
            "sumstats_file": sumstats_config[trait_name],
            "hdf5_path": config.hdf5_path,
            "annotation": config.annotation,

            "num_processes": config.max_processes,
            "ldscore_dir": ldscore_config.ldscore_save_dir,
            "w_file": config.w_file,
            "gtf_annotation_file": config.gtffile,
            "bfile_root": config.bfile_root,
            "keep_snp_root": config.keep_snp_root,
            # "mkscore_feather_file": latent_to_gene_config.output_feather_path,
            "spatial_ldsc_save_dir": config.ldsc_save_dir,
            "cauchy_dir": f"{config.workdir}/{config.sample_name}/cauchy_combination",
            'visualize_dir': f"{config.workdir}/{config.sample_name}/visualize",
            "diagnosis_dir": f"{config.workdir}/{config.sample_name}/diagnosis",

            "Spending_time": format_duration(time.time() - pipeline_start_time),
            "Finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        # Pass the run parameter dictionary to the report generation function
        run_report(report_config, run_parameters=run_parameter_dict)

    logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    # Example usage:
    # config = RunAllModeConfig(
    #     workdir='/mnt/e/0_Wenhao/7_Projects/20231213_GPS_Liyang/test/20240902_gsMap_Local_Test/0908_workdir_test',
    #     sample_name='Human_Cortex_151507',
    #     gsMap_resource_dir='/mnt/e/0_Wenhao/7_Projects/20231213_GPS_Liyang/test/20240902_gsMap_Local_Test/gsMap_resource',
    #     hdf5_path='/mnt/e/0_Wenhao/7_Projects/20231213_GPS_Liyang/test/20240902_gsMap_Local_Test/example_data/ST/Cortex_151507.h5ad',
    #     annotation='layer_guess',
    #     data_layer='count',
    #     trait_name='Depression_2023_NatureMed',
    #     sumstats_file='/mnt/e/0_Wenhao/7_Projects/20231213_GPS_Liyang/test/20240902_gsMap_Local_Test/example_data/GWAS/Depression_2023_NatureMed.sumstats.gz',
    #     homolog_file=None,
    #     max_processes=10
    # )

    path_prefix = '/storage/yangjianLab/chenwenhao/projects/202312_GPS/'
    config = RunAllModeConfig(
        workdir=('%s/test/20240902_gsMap_Local_Test/0908_workdir_test' % path_prefix),
        sample_name='Human_Cortex_151507',
        gsMap_resource_dir=('%s/test/20240902_gsMap_Local_Test/gsMap_resource' % path_prefix),
        hdf5_path=('%s/test/20240902_gsMap_Local_Test/example_data/ST/Cortex_151507.h5ad' % path_prefix),
        annotation='layer_guess',
        data_layer='count',
        trait_name='Depression_2023_NatureMed',
        sumstats_file=(
                '%s/test/20240902_gsMap_Local_Test/example_data/GWAS/Depression_2023_NatureMed.sumstats.gz' % path_prefix),
        homolog_file=None,
        max_processes=10
    )
    #
    # config = RunAllModeConfig(
    #     workdir=('%s/test/20240902_gsMap_Local_Test/0908_workdir_test' % path_prefix),
    #     sample_name='E16.5_E1S1.MOSTA',
    #     gsMap_resource_dir=('%s/test/20240902_gsMap_Local_Test/gsMap_resource' % path_prefix),
    #     hdf5_path='/storage/yangjianLab/songliyang/SpatialData/Data/Embryo/Mice/Cell_MOSTA/h5ad/E16.5_E1S1.MOSTA.h5ad',
    #     annotation='annotation',
    #     data_layer='count',
    #     trait_name='Depression_2023_NatureMed',
    #     sumstats_file=(
    #                 '%s/test/20240902_gsMap_Local_Test/example_data/GWAS/Depression_2023_NatureMed.sumstats.gz' % path_prefix),
    #     homolog_file='/storage/yangjianLab/chenwenhao/projects/202312_GPS/test/20240902_gsMap_Local_Test/gsMap_resource/homologs/mouse_human_homologs.txt',
    #     max_processes=10
    # )

    # config = RunAllModeConfig(
    #     workdir='/storage/yangjianLab/chenwenhao/projects/202312_GPS/test/20240817_vanilla_pipeline_mouse_embryo_v4/E16.5_E1S1.MOSTA',
    #     sample_name='E16.5_E1S1.MOSTA',
    #     gsMap_resource_dir='/storage/yangjianLab/songliyang/SpatialData/gsMap_resource',
    #     hdf5_path='/storage/yangjianLab/songliyang/SpatialData/Data/Embryo/Mice/Cell_MOSTA/h5ad/E16.5_E1S1.MOSTA.h5ad',
    #     annotation='layer_guess',
    #     trait_name='Depression_2023_NatureMed',
    #     sumstats_config_file='/storage/yangjianLab/songliyang/SpatialData/Data/Embryo/Mice/Cell_MOSTA/ldsc_enrichment_frac/E16.5_E1S1/sumstats_config.yaml',
    #     homolog_file=None,
    #     max_processes=10
    # )
    run_pipeline(config)
