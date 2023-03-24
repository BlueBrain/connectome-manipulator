{
  "circuit_path": "",
  "circuit_config": "circuit_100.json",
  "output_path": "",
  "seed": 3210,
  "N_split_nodes": 1,
  "manip": {
    "name": "ConnWiring_DD",
    "fcts": [
      {
        "source": "conn_wiring",
        "kwargs": {
          "amount_pct": 100.0,
          "prob_model_file": {
            "model": "ConnProb2ndOrderExpModel",
            "scale": 0.1488091516112886,
            "exponent": 0.007560220091714113
          },
          "nsynconn_model_file": {
            "model": "ConnPropsModel",
            "src_types": [
              "GEN_mtype"
            ],
            "tgt_types": [
              "GEN_mtype"
            ],
            "prop_stats": {
              "n_syn_per_conn": {
                "GEN_mtype": {
                  "GEN_mtype": {
                    "type": "gamma",
                    "mean": 3.0,
                    "std": 1.5,
                    "dtype": "int",
                    "lower_bound": 1,
                    "upper_bound": 1000
                  }
                }
              }
            }
          },
          "delay_model_file": {
            "model": "LinDelayModel",
            "delay_mean_coefs": [
              0.75,
              0.003
            ],
            "delay_std": 0.5,
            "delay_min": 0.2
          },
          "morph_ext": "swc"
        }
      }
    ]
  }
}