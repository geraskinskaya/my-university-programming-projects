# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


python3 scripts/skeleton_script_baseline_model.py data/HGVS_2020_small_VEP_baseline.tsv data/BLOSUM62.txt -o data/HGVS_2020_small_baseline_scores.tsv


python3 scripts/skeleton_script_create_roc_plot.py -ibench data/HGVS_2020_small_benchmark.tsv -ipred data/HGVS_2020_small_polyphen_scores.tsv -color -o output/ROCplot_HGVS_2020_small_polyphen.png

python3 scripts/skeleton_script_create_roc_plot.py -ibench data/HGVS_2020_small_benchmark.tsv -ipred data/HGVS_2020_small_sift_scores.tsv -color -o output/ROCplot_HGVS_2020_small_sift.png

python3 scripts/skeleton_script_create_roc_plot.py -ibench data/HGVS_2020_small_benchmark.tsv -ipred data/HGVS_2020_small_baseline_scores.tsv -color -o output/ROCplot_HGVS_2020_small_baseline.png


python3 scripts/skeleton_script_create_roc_plot.py -ibench data/HGVS_2020_small_benchmark.tsv -ipred data/HGVS_2020_small_baseline_scores.tsv -color -o output/ROCplot_HGVS_2020_small_baseline.png

python3 scripts/skeleton_script_create_roc_plot.py -ipred data/HGVS_2020_small_polyphen_scores.tsv -ipred data/HGVS_2020_small_sift_scores.tsv -ipred data/HGVS_2020_small_baseline_scores.tsv -ibench data/HGVS_2020_small_benchmark.tsv -o output/ROCplot_all.png
