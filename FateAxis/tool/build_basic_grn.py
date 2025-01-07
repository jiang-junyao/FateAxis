# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:04:56 2025

@author: junyao
"""

from celloracle import motif_analysis as ma

def build_basic_grn(peak,ref_genome,
                    fpr_thr=0.02,ncores=10):
    peak = peak.replace({':': '_', '-': '_'}, regex=True)
    tfi = ma.TFinfo(peak_data_frame=peak,
                ref_genome=ref_genome,
                genomes_dir=None)
    tfi.scan(fpr=fpr_thr,
             motifs=None, 
             verbose=True)
    tfi.reset_filtering()
    tfi.filter_motifs_by_score(threshold=ncores)
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)
    df = tfi.to_dataframe()
    return df