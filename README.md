## README

The whole repository can be cloned/downloaded as a zip file (see also [OSF][1]). The code is written in Python and relies mainly on NumPy, SciPy, and pandas (+Matplotlib for visualizations). A Go implementation with basic functionality is also provided.

Objective metrics were computed using pyiqa (as outlined in the scripts inside the **Python/pyiqa_scripts** folder. For more information about pyiqa, see:

    @misc{pyiqa,
      title={{IQA-PyTorch}: PyTorch Toolbox for Image Quality Assessment},
      author={Chaofeng Chen and Jiadi Mo},
      year={2022},
      howpublished = "[Online]. Available: \url{https://github.com/chaofengc/IQA-PyTorch}"
    }
    
The results of the objective metrics can be found in the **../metric_values** folders. The example code leverages data from CID 2013 (see [Zenodo][2] and the [home page of the research group][3]).

### Citation

If you found the implementation useful, please cite:

    @Article{Zizien_2025,
      author    = {Zizien, Adam and Fliegel, Karel},
      journal   = {IEEE Access},
      title     = {Correlating Without Confidence: The Overlooked Role of Uncertainty when Ranking Objective Measures},
      year      = {2025},
      issn      = {2169-3536},
      pages     = {1--1},
      doi       = {10.1109/access.2025.3544307},
      publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
    }


  [1]: https://osf.io/2x4g8/ "OSF"
  [2]: https://zenodo.org/records/2647033 "CID 2013 on Zenodo"
  [3]: https://researchportal.helsinki.fi/en/publications/cid2013-a-database-for-evaluating-no-reference-image-quality-asse "CID 2013 research group home page"
