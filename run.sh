MODELDIR=/data/phalendj/kaggle/rsna2024/Submission_20240905
DATADIR=/data/phalendj/kaggle/rsna2024
IMGDIR=train_images
# poetry run python rsna2024/run.py --config-path=${MODELDIR}/Instance/.hydra/ --config-name=config.yaml mode=test train=False directories.relative_directory=${DATADIR}/ directories.image_directory=${IMGDIR} load_directory=${MODELDIR}/Instance/ training.workers=4 training.batch_size=4  clean=0
# peotry run python rsna2024/run.py --config-path=${MODELDIR}/Segment/.hydra/ --config-name=config.yaml mode=test train=False directories.relative_directory=${DATADIR}/ directories.image_directory=${IMGDIR} load_directory=${MODELDIR}/Segment/ dataset.center_file=predicted_label_coordinates.csv training.workers=4 training.batch_size=4 clean=0
poetry run python rsna2024/run.py --config-path=${MODELDIR}/Spinal/.hydra/ --config-name=config.yaml train=False result=predict directories.relative_directory=${DATADIR}/ directories.image_directory=${IMGDIR} load_directory=${MODELDIR}/Spinal/ dataset.center_file=${MODELDIR}/Segment/all_predicted_center_coordinates.csv training.workers=12 training.batch_size=12
mv submission.csv submission_spinal.csv
poetry run python rsna2024/run.py --config-path=${MODELDIR}/Foraminal/.hydra/ --config-name=config.yaml train=False result=predict directories.relative_directory=${DATADIR}/ directories.image_directory=${IMGDIR} load_directory=${MODELDIR}/Foraminal/ dataset.center_file=${MODELDIR}/Segment/all_predicted_center_coordinates.csv training.workers=12 training.batch_size=12
mv submission.csv submission_foraminal.csv
poetry run python rsna2024/run.py --config-path=${MODELDIR}/Subarticular/.hydra/ --config-name=config.yaml train=False result=predict directories.relative_directory=${DATADIR}/ directories.image_directory=${IMGDIR} load_directory=${MODELDIR}/Subarticular/ dataset.center_file=${MODELDIR}/Segment/all_predicted_center_coordinates.csv training.workers=12 training.batch_size=12
mv submission.csv submission_subarticular.csv
