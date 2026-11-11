# Third-party content

| Component | Used for | Licence | In this repo? |
|---|---|---|---|
| NVIDIA Isaac Sim 6.1 (pip `isaacsim`) | simulator, renderer, physics | NVIDIA Omniverse / Isaac Sim licence and EULA (you accept it on your machine) | no: installed by `scripts/setup.sh` |
| Isaac Sim content: `Environments/Hospital/hospital.usd`, `Robots/Clearpath/RidgebackFranka`, `People/Characters`, people animations | hospital shell, robot, walking staff | NVIDIA content licence | no: downloaded at run time from NVIDIA's content server by your Isaac Sim |
| NVIDIA SimReady assets (`nvidia/simready-assets` on Hugging Face, pinned commit `653d91e182011f357cf1644013cc94ddf194d349`) | beds, trolleys, instruments, shelves, boxes inside the six rooms | CC BY 4.0, attribution to NVIDIA | composed room files are in the separate room archive; the licence text is each asset's `.metadata/LICENSE.txt` in the source dataset |
| NVIDIA Isaac Lab-Arena (optional) | only to rebuild the room files | see its repository | no |
| numpy, SciPy (BSD), Pillow (HPND), OpenCV (Apache-2.0), onnxruntime (MIT), rapidocr-onnxruntime (Apache-2.0) | image handling, the information-only scripted robot | their own licences | no: installed by pip |
| Provider APIs (Anthropic, OpenAI, Google) | model calls | provider terms; you bring your own key | no |

The code in this repository (everything outside the rows above) is MIT licensed, see [LICENSE](LICENSE). Textures in `bench/hospital/h1_textures` and `bench/hospital/t1_textures` are drawn by the code of this repository.
