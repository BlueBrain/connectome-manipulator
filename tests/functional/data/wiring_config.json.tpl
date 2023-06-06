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
        "amount_pct": 100.0,
        "morph_ext": "swc",
        "model_config": {
          "prob_model_spec": {
            "model": "ConnProb2ndOrderExpModel",
            "scale": 0.1488091516112886,
            "exponent": 0.007560220091714113
          },
          "nsynconn_model_spec": {
            "model": "NSynConnModel",
            "mean": 3.0,
            "std": 1.5
          },
          "delay_model_spec": {
            "model": "LinDelayModel",
            "delay_mean_coeff_a": 0.75,
            "delay_mean_coeff_b": 0.003,
            "delay_std": 0.5,
            "delay_min": 0.2
          }
        }
      }
    ]
  }
}
