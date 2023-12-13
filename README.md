# GPS

# How to use:
## STEP-1: Find the latent representations 
root=/storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/processed/h5ad

ls ${root} | while read file
do
   name=($(echo ${file} | cut -d'.' -f 1)) 

   command="python3 /storage/yangjianLab/songliyang/SpatialData/spatial_ldsc_v1/Find_Latent_Representations.py \
   --spe_path ${root} \
   --spe_name ${file} \
   --annotation layer_guess \
   --type count \
   --spe_out /storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/${name}/h5ad"

    qsubshcom_gpu "$command" 1 50G GAT_${name} 2:00:00 "--qos gpu-huge -queue=v100,a40-tmp,a40-quad"
done


## STEP-2: Find marker genes
root=/storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/processed/h5ad
ls ${root} | grep h5ad | while read file
do
	name=($(echo ${file} | cut -d'.' -f 1))

	command="python3 /storage/yangjianLab/songliyang/SpatialData/spatial_ldsc_v1/Latent_to_Gene_V2.py \
	--latent_representation latent_GVAE \
	--spe_path /storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/${name}/h5ad \
	--spe_name ${name}_add_latent.h5ad \
	--num_processes 4 \
	--type count \
	--annotation layer_guess \
	--num_neighbour 51 \
	--spe_out /storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/${name}/gene_markers"

	qsubshcom "$command" 4 50G mkS_${name} 24:00:00 "--qos huge -queue=intel-sc3,amd-ep2,amd-ep2-short"
done



## STEP-3: markers to SNP annotations
root=/storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/processed/h5ad
ls ${root} | grep h5ad | while read file
do
	name=($(echo ${file} | cut -d'.' -f 1))

	command="python3 /storage/yangjianLab/songliyang/SpatialData/spatial_ldsc_v1/Make_Annotations_V2.py \
	--mk_score_file /storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/${name}/gene_markers/${name}_rank.feather \
	--gtf_file /storage/yangjianLab/songliyang/ReferenceGenome/GRCh37/gencode.v39lift37.annotation.gtf \
	--bfile_root /storage/yangjianLab/sharedata/LDSC_resource/1000G_EUR_Phase3_plink/1000G.EUR.QC \
	--annot_root /storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/${name}/snp_annotation \
	--keep_snp /storage/yangjianLab/sharedata/LDSC_resource/hapmap3_snps/hm \
	--annot_name ${name} \
	--const_max_size 500 \
	--chr {TASK_ID} \
	--ld_wind_cm 1"

	qsubshcom "$command" 5 60G annS_${name} 24:00:00 "-array=1-22 --qos huge -queue=intel-sc3,amd-ep2,amd-ep2-short"
done









