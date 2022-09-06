# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import torch
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from pt.learners.mammo_learner import MammoLearner, load_datalist, MockClientEngine


def inference(model_filepath, dataset_root, datalist_prefix, output_root):
    print("MammoLearner inference...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        aggregation_epochs=1,
        lr=1e-2,
    )
    engine = MockClientEngine()
    fl_ctx = engine.fl_ctx_mgr.new_context()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, "/tmp/debug")

    print("test initialize...")
    learner.initialize(parts={}, fl_ctx=fl_ctx)

    ckpt = torch.load(model_filepath, map_location=device)
    learner.model.load_state_dict(ckpt["model"])
    print(f"Restored weights from {model_filepath}.")

    print("test...")
    probs = learner.local_valid(
        valid_loader=learner.test_loader, abort_signal=Signal(), return_probs_only=True
    )
    print(f"Created {len(probs)} probs on `test` set.")

    val_results = {
        "test_site": {
            "SRV_best_FL_global_model.pt": {
                "test_probs": probs
            }
        }

    }

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    with open(os.path.join(output_root, "test_predictions.json"), "w") as f:
        json.dump(val_results, f)

    print("finished testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filepath", type=str, default="/result_server/run_1/app_server/best_FL_global_model.pt", help="model checkpoint")
    parser.add_argument("--dataset_root", type=str, default="/data/preprocessed", help="dataset root")
    parser.add_argument("--datalist_prefix", type=str, default="/data/dataset_blinded_", help="datalist prefix")
    parser.add_argument("--output_root", type=str, default="/output", help="output directory for predictions")
    args = parser.parse_args()

    assert "best_FL_global_model.pt" in args.model_filepath, f"Challenge assumes inference model to be `best_FL_global_model.pt`"

    inference(
        model_filepath=args.model_filepath,
        dataset_root=args.dataset_root,
        datalist_prefix=args.datalist_prefix,
        output_root=args.output_root
    )
