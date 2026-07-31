# State test PDFs that were downloaded

The PDFs themselves are **not kept**: roughly 650MB of state-copyright
material, of no further use once items are extracted, and they must never
enter this public repo. This is the record of what was fetched.

Re-download with the command listed for each state. Checksums are the
first 16 hex characters of the sha256, enough to spot a changed file.

**Total: 409 PDFs, 653 MB.**

## Texas STAAR (`tx`)

tea.texas.gov released test questions (2018-2019 only; earlier years removed, 2021+ has no answer keys, 2022+ is online-only)

- **58 PDFs**, 61 MB, yielding **122 extracted items**
- rebuild: `python src/staar_extract.py`

| file | MB | sha256[:16] |
|---|---|---|
| `2018-staar-3-math-key.pdf` | 0.0 | `ae1df70e337cd2d2` |
| `2018-staar-3-math-test.pdf` | 1.1 | `e04071a432b07d9a` |
| `2018-staar-3-reading-key.pdf` | 0.0 | `aa4b5df71a31c5d4` |
| `2018-staar-4-math-key.pdf` | 0.0 | `9a7733044dc96e51` |
| `2018-staar-4-math-test.pdf` | 1.3 | `fa0cac25c59007f6` |
| `2018-staar-4-reading-key.pdf` | 0.0 | `2903309eb7d27c22` |
| `2018-staar-4-reading-test.pdf` | 2.7 | `fccae5bea09719ba` |
| `2018-staar-5-math-key.pdf` | 0.0 | `3d1cbb8370947230` |
| `2018-staar-5-math-test.pdf` | 1.2 | `9d2b31c909a9fb33` |
| `2018-staar-5-reading-key.pdf` | 0.0 | `fac6cac39f69bf99` |
| `2018-staar-5-reading-test.pdf` | 2.3 | `d65a7ce8c779bf30` |
| `2018-staar-5-science-key.pdf` | 0.2 | `772a879f760b4a0d` |
| `2018-staar-5-science-test.pdf` | 2.2 | `fb03905445551f3c` |
| `2018-staar-6-math-key.pdf` | 0.0 | `2adaf4a39e884f2d` |
| `2018-staar-6-math-test.pdf` | 0.7 | `fb38e3b58a1c18b4` |
| `2018-staar-6-reading-key.pdf` | 0.0 | `d594a88a7550701c` |
| `2018-staar-6-reading-test.pdf` | 1.2 | `cf3d5a8f3e6767b1` |
| `2018-staar-7-math-key.pdf` | 0.0 | `85c02afb2abdb0fa` |
| `2018-staar-7-math-test.pdf` | 0.9 | `3c37f0270c592235` |
| `2018-staar-7-reading-key.pdf` | 0.0 | `381347db65f9892c` |
| `2018-staar-7-reading-test.pdf` | 1.8 | `54b76f87b324c955` |
| `2018-staar-8-math-key.pdf` | 0.0 | `936b41de2f5f8167` |
| `2018-staar-8-math-test.pdf` | 1.2 | `924b7f38c3eea13a` |
| `2018-staar-8-reading-key.pdf` | 0.0 | `08aeff2466227cc4` |
| `2018-staar-8-reading-test.pdf` | 2.6 | `4c9538ec1274905b` |
| `2018-staar-8-science-key.pdf` | 0.2 | `9206db987526ef45` |
| `2018-staar-8-science-test.pdf` | 1.6 | `6a23b5b660d5d84a` |
| `2018-staar-8-social-studies-key.pdf` | 0.0 | `96fb27ace187a76b` |
| `2018-staar-8-social-studies-test.pdf` | 5.0 | `e83ac8eedb7966fa` |
| `2019-staar-3-math-key.pdf` | 0.1 | `fd8e08ba6ae4c852` |
| `2019-staar-3-math-test.pdf` | 0.5 | `584aa26f567c7e84` |
| `2019-staar-3-reading-test.pdf` | 1.9 | `70338f1721667aed` |
| `2019-staar-4-math-key.pdf` | 0.1 | `affd81237dfa056a` |
| `2019-staar-4-math-test.pdf` | 0.3 | `e70be526beda3f65` |
| `2019-staar-4-reading-key.pdf` | 0.1 | `74fe7852c22ab8d6` |
| `2019-staar-4-reading-test.pdf` | 3.6 | `eb5dad3fc5b64630` |
| `2019-staar-5-math-key.pdf` | 0.1 | `31a1e186f3ae3663` |
| `2019-staar-5-math-test.pdf` | 0.2 | `f6f177467953250b` |
| `2019-staar-5-reading-key.pdf` | 0.1 | `35c374e419e3d4d7` |
| `2019-staar-5-reading-test.pdf` | 1.2 | `642292640a20d836` |
| `2019-staar-5-science-key.pdf` | 0.1 | `663fa01fafa58a67` |
| `2019-staar-5-science-test.pdf` | 3.6 | `41d2fc2384939cc2` |
| `2019-staar-6-math-key.pdf` | 0.1 | `06570abb4c57bd21` |
| `2019-staar-6-math-test.pdf` | 0.5 | `82a7fa780284a13a` |
| `2019-staar-6-reading-key.pdf` | 0.1 | `4d1b57950d8d7787` |
| `2019-staar-6-reading-test.pdf` | 1.7 | `e85838fa600513f6` |
| `2019-staar-7-math-key.pdf` | 0.1 | `5b1f7709257ab457` |
| `2019-staar-7-math-test.pdf` | 3.8 | `a6e682d21cbaf074` |
| `2019-staar-7-reading-key.pdf` | 0.1 | `cb85a55316634cb9` |
| `2019-staar-7-reading-test.pdf` | 1.4 | `2bb559aa2b255bf8` |
| `2019-staar-8-math-key.pdf` | 0.1 | `4832188678f1c5cf` |
| `2019-staar-8-math-test.pdf` | 0.5 | `e744532890780f5f` |
| `2019-staar-8-reading-key.pdf` | 0.1 | `b475ae3da4ec2ce1` |
| `2019-staar-8-reading-test.pdf` | 6.1 | `b1d2124907c4286b` |
| `2019-staar-8-science-key.pdf` | 0.1 | `dde99d0c5964efda` |
| `2019-staar-8-science-test.pdf` | 7.0 | `049beeba4fdcdd9d` |
| `2019-staar-8-social-studies-answer-key.pdf` | 0.1 | `66a1f6b1b1a57faa` |
| `2019-staar-8-social-studies-test.pdf` | 0.8 | `e92b8f0459e8fcb6` |

## Pennsylvania PSSA (`pa`)

pa.gov item and scoring samplers; answer key, DOK, p-values and per-option rationales are inline with each item

- **109 PDFs**, 293 MB, yielding **235 extracted items**
- rebuild: `python src/extract_pa.py`

| file | MB | sha256[:16] |
|---|---|---|
| `2015-pssa-ela-grade-3.pdf` | 5.7 | `1926b4abeefdbfeb` |
| `2015-pssa-ela-grade-4.pdf` | 9.2 | `c438573a4bf737c9` |
| `2015-pssa-ela-grade-5.pdf` | 10.1 | `c2d40f7a8f5b3266` |
| `2015-pssa-ela-grade-6.pdf` | 2.7 | `e4e2e27cb67be7f8` |
| `2015-pssa-ela-grade-7.pdf` | 2.9 | `11bc1b65a923fbf4` |
| `2015-pssa-ela-grade-8.pdf` | 3.0 | `1bc779f4f93c9540` |
| `2015-pssa-math-grade-3.pdf` | 7.7 | `810b179500cc38b9` |
| `2015-pssa-math-grade-4.pdf` | 5.6 | `ea2e4213f90a7b35` |
| `2015-pssa-math-grade-5.pdf` | 5.1 | `a5339e16b431c8a3` |
| `2015-pssa-math-grade-6.pdf` | 7.5 | `dd328701760afaf5` |
| `2015-pssa-math-grade-7.pdf` | 6.1 | `14f9ad435b88eb35` |
| `2015-pssa-math-grade-8.pdf` | 10.6 | `48ca21e934e1677c` |
| `2016-pssa-ela-grade-3.pdf` | 3.6 | `a3e9064efe8614ad` |
| `2016-pssa-ela-grade-4.pdf` | 4.6 | `dd448d18b0da2e2a` |
| `2016-pssa-ela-grade-5.pdf` | 5.2 | `ba5447b30ef7fd4c` |
| `2016-pssa-ela-grade-6.pdf` | 5.6 | `46707764744535ca` |
| `2016-pssa-ela-grade-7.pdf` | 5.0 | `a87e5d3771b6d1c0` |
| `2016-pssa-ela-grade-8.pdf` | 5.5 | `18bb8599c3215799` |
| `2016-pssa-math-grade-3.pdf` | 3.3 | `401613a3aaebc9a5` |
| `2016-pssa-math-grade-4.pdf` | 3.5 | `5199baf70ff4b058` |
| `2016-pssa-math-grade-5.pdf` | 3.5 | `be2ccf521b9d5a30` |
| `2016-pssa-math-grade-6.pdf` | 4.1 | `df85adea6116e27a` |
| `2016-pssa-math-grade-7.pdf` | 4.0 | `ed50f005ebfbb8a3` |
| `2016-pssa-math-grade-8.pdf` | 4.3 | `bd6ba526aa9dba21` |
| `2016-pssa-science-grade-4.pdf` | 3.5 | `969bac69ce2569ca` |
| `2016-pssa-science-grade-8.pdf` | 3.5 | `56390a9740a8eb7f` |
| `2018-pssa-ela-grade-3.pdf` | 1.5 | `e50eb71faeea6ff3` |
| `2018-pssa-ela-grade-4.pdf` | 2.1 | `7d63aae42c6fb0dd` |
| `2018-pssa-ela-grade-5.pdf` | 2.1 | `bf9f5aa779f21573` |
| `2018-pssa-ela-grade-6.pdf` | 2.2 | `d36df3f099f02573` |
| `2018-pssa-ela-grade-7.pdf` | 2.4 | `d41f62cb16d81266` |
| `2018-pssa-ela-grade-8.pdf` | 2.4 | `b0d86e04564e6214` |
| `2018-pssa-math-grade-3.pdf` | 2.6 | `55ab6cae0196cb93` |
| `2018-pssa-math-grade-4.pdf` | 2.9 | `faf943975898e459` |
| `2018-pssa-math-grade-5.pdf` | 2.9 | `51b521c8540e5d73` |
| `2018-pssa-math-grade-6.pdf` | 2.7 | `90f17bc08a0441fe` |
| `2018-pssa-math-grade-7.pdf` | 3.6 | `4c146006e0a428c9` |
| `2018-pssa-math-grade-8.pdf` | 3.2 | `33a4b0bdbd3ef0da` |
| `2018-pssa-science-grade-4.pdf` | 3.2 | `d014cac6f4c32003` |
| `2018-pssa-science-grade-8.pdf` | 3.0 | `17b83cec980d862c` |
| `2019-pssa-ela-grade-3.pdf` | 1.9 | `9a391c6cf19d72dd` |
| `2019-pssa-ela-grade-4.pdf` | 2.4 | `536ea770e8625c28` |
| `2019-pssa-ela-grade-5.pdf` | 2.7 | `23e83c80f950ad51` |
| `2019-pssa-ela-grade-6.pdf` | 2.7 | `04c67c089eb091a8` |
| `2019-pssa-ela-grade-7.pdf` | 2.2 | `b233f8a0a23aae76` |
| `2019-pssa-ela-grade-8.pdf` | 2.3 | `e23adbbafb05b46a` |
| `2019-pssa-math-grade-3.pdf` | 2.7 | `e8f6b652f63045e8` |
| `2019-pssa-math-grade-4.pdf` | 2.6 | `e664824f55e5e2b1` |
| `2019-pssa-math-grade-5.pdf` | 2.5 | `cdeb02b551ff0981` |
| `2019-pssa-math-grade-6.pdf` | 2.3 | `89f0fa420aea4e9c` |
| `2019-pssa-math-grade-7.pdf` | 2.5 | `3b501bd679a0c4f5` |
| `2019-pssa-math-grade-8.pdf` | 3.2 | `082e1c2584004222` |
| `2019-pssa-science-grade-4.pdf` | 2.8 | `fede13a44704a668` |
| `2019-pssa-science-grade-8.pdf` | 2.6 | `877cccf837ac6598` |
| `2021-pssa-ela-grade-3.pdf` | 0.7 | `d56fee1cd986abab` |
| `2021-pssa-ela-grade-4.pdf` | 0.8 | `12ee8196eef65315` |
| `2021-pssa-ela-grade-5.pdf` | 1.0 | `189a253968adfc2b` |
| `2021-pssa-ela-grade-6.pdf` | 1.0 | `fd5e7c2ad39c6125` |
| `2021-pssa-ela-grade-7.pdf` | 0.9 | `911a3bfd6da02cee` |
| `2021-pssa-ela-grade-8.pdf` | 1.1 | `7845572e1da89f69` |
| `2021-pssa-math-grade-3.pdf` | 2.3 | `69bb4ece1ea6dedd` |
| `2021-pssa-math-grade-4.pdf` | 1.3 | `ab9460d518e7a85e` |
| `2021-pssa-math-grade-5.pdf` | 1.7 | `6ff35d2e17c40b32` |
| `2021-pssa-math-grade-6.pdf` | 1.9 | `0824fe2adbfbb20d` |
| `2021-pssa-math-grade-7.pdf` | 1.6 | `587fdef84c3a5e08` |
| `2021-pssa-math-grade-8.pdf` | 1.9 | `c9c4fe5f3c6d8537` |
| `2021-pssa-science-grade-4.pdf` | 2.1 | `e59570971d029867` |
| `2021-pssa-science-grade-8.pdf` | 1.9 | `e77c95af5a12baae` |
| `2022-pssa-ela-grade-3.pdf` | 0.6 | `00598b747b4d0803` |
| `2022-pssa-ela-grade-4.pdf` | 0.7 | `cdb414a7958b0b34` |
| `2022-pssa-ela-grade-5.pdf` | 0.7 | `656b222e2a7e357f` |
| `2022-pssa-ela-grade-6.pdf` | 0.7 | `baab4e20e8758094` |
| `2022-pssa-ela-grade-7.pdf` | 0.7 | `cab2bc62b3d71189` |
| `2022-pssa-ela-grade-8.pdf` | 0.8 | `968802ddf71eecc8` |
| `2022-pssa-math-grade-3.pdf` | 1.5 | `63b9e221682a25c9` |
| `2022-pssa-math-grade-4.pdf` | 1.3 | `21f4cf7f4bde2d98` |
| `2022-pssa-math-grade-5.pdf` | 1.2 | `bf006849915ddb76` |
| `2022-pssa-math-grade-6.pdf` | 1.3 | `9e7606cd35ed5725` |
| `2022-pssa-math-grade-7.pdf` | 1.2 | `8a9a88f9ba348ecc` |
| `2022-pssa-math-grade-8.pdf` | 1.2 | `e183f70af3dadbf8` |
| `2022-pssa-science-grade-4.pdf` | 0.9 | `7d20242ce673746e` |
| `2022-pssa-science-grade-8.pdf` | 0.8 | `e1d13f531aac97ff` |
| `2023-pssa-ela-grade-3.pdf` | 0.6 | `74111e3369c4d831` |
| `2023-pssa-ela-grade-4.pdf` | 0.6 | `f7597f464f670d05` |
| `2023-pssa-ela-grade-5.pdf` | 0.7 | `5ea508fc3974de4c` |
| `2023-pssa-ela-grade-6.pdf` | 0.6 | `febda847f221800c` |
| `2023-pssa-ela-grade-7.pdf` | 0.7 | `cabc9546d5699dbc` |
| `2023-pssa-ela-grade-8.pdf` | 0.7 | `6732dc0d481124d9` |
| `2023-pssa-math-grade-3.pdf` | 2.1 | `7d5ffd669c50190f` |
| `2023-pssa-math-grade-4.pdf` | 1.9 | `35b5e15362a1c43c` |
| `2023-pssa-math-grade-5.pdf` | 2.8 | `695b3864b9980406` |
| `2023-pssa-math-grade-6.pdf` | 2.5 | `728ddca1e8d404ea` |
| `2023-pssa-math-grade-7.pdf` | 2.2 | `877aca65cd0ff6a2` |
| `2023-pssa-math-grade-8.pdf` | 2.8 | `2dd8d9972401ea54` |
| `2023-pssa-science-grade-4.pdf` | 1.0 | `13f62f2b71982016` |
| `2023-pssa-science-grade-8.pdf` | 1.0 | `66d0466991e64e41` |
| `2024-pssa-ela-grade-3.pdf` | 1.2 | `2df0930e93367187` |
| `2024-pssa-ela-grade-4.pdf` | 1.6 | `0feedbcb3249c6b8` |
| `2024-pssa-ela-grade-5.pdf` | 1.5 | `243cb2b88aee68d0` |
| `2024-pssa-ela-grade-6.pdf` | 1.6 | `e86cc42e48aaa33c` |
| `2024-pssa-ela-grade-7.pdf` | 1.7 | `f17072739dd9efb7` |
| `2024-pssa-ela-grade-8.pdf` | 1.8 | `99a7658dbbb12d11` |
| `2024-pssa-math-grade-3.pdf` | 3.6 | `a454acc9edf19e4f` |
| `2024-pssa-math-grade-4.pdf` | 2.9 | `bd8e3c458bf717c6` |
| `2024-pssa-math-grade-5.pdf` | 3.3 | `a5b9a73d69da01ba` |
| `2024-pssa-math-grade-6.pdf` | 2.6 | `59ea0ad3f241a8f1` |
| `2024-pssa-math-grade-7.pdf` | 3.1 | `c9ed02913d438e17` |
| `2024-pssa-math-grade-8.pdf` | 4.1 | `c7b4db5239c3fb06` |
| `2024-pssa-science-grade-8.pdf` | 2.6 | `ba9edabe0c22bcea` |

## California CST (`ca`)

released test questions via the Wayback Machine; removed from cde.ca.gov and the live site is behind a captcha

- **101 PDFs**, 70 MB, yielding **375 extracted items**
- rebuild: `python src/extract_ca.py`

| file | MB | sha256[:16] |
|---|---|---|
| `css05rtqalg1math.pdf` | 0.1 | `20c690acd11b7f39` |
| `css05rtqalg2math.pdf` | 0.1 | `cb3142d8dec118fd` |
| `css05rtqbio.pdf` | 0.1 | `aefa5ac3724f613f` |
| `css05rtqchem.pdf` | 0.1 | `bc0e822180bb9fb0` |
| `css05rtqearthsci.pdf` | 0.2 | `de5965d01a5572d9` |
| `css05rtqgeomath.pdf` | 0.2 | `9fea62adf4e84377` |
| `css05rtqgr10ela.pdf` | 0.8 | `3abffcdfb3f68073` |
| `css05rtqgr10hist.pdf` | 0.1 | `866367f44fa5ee67` |
| `css05rtqgr11ela.pdf` | 0.2 | `fcf9b4a47d86cae0` |
| `css05rtqgr2ela.pdf` | 0.1 | `fd75043198066265` |
| `css05rtqgr2math.pdf` | 0.9 | `6ff0d8e8a97e0c36` |
| `css05rtqgr3ela.pdf` | 0.4 | `1ee8583c2576027f` |
| `css05rtqgr3math.pdf` | 0.2 | `adfb5ad6248dbbd8` |
| `css05rtqgr4ela.pdf` | 0.4 | `88a92da6773dd922` |
| `css05rtqgr4math.pdf` | 0.1 | `b2a115b3a302b0f1` |
| `css05rtqgr5ela.pdf` | 0.2 | `f1d4c32c090456e0` |
| `css05rtqgr5math.pdf` | 0.2 | `edf318feb491d77b` |
| `css05rtqgr5sci.pdf` | 0.1 | `8a7b7e581f0f4145` |
| `css05rtqgr68hist.pdf` | 0.3 | `4e6576a7a4151a42` |
| `css05rtqgr6ela.pdf` | 0.3 | `78b1ffff01f31539` |
| `css05rtqgr6math.pdf` | 0.2 | `4c1f3da4ac3c5087` |
| `css05rtqgr7ela.pdf` | 0.2 | `7f705adcb612eccf` |
| `css05rtqgr7math.pdf` | 0.2 | `5024984e23da5e0c` |
| `css05rtqgr8ela.pdf` | 0.3 | `58da0bad4e96d294` |
| `css05rtqgr9ela.pdf` | 0.4 | `2328a5b4ff4dc7d9` |
| `css05rtqhistgr11.pdf` | 0.2 | `ab4efdd165ad5836` |
| `css05rtqphysics.pdf` | 0.1 | `d5a86ec29c286117` |
| `cst04rtqelagr2.pdf` | 0.1 | `141a90eeee920d34` |
| `cst04rtqelagr3.pdf` | 0.1 | `6b19778ced96acba` |
| `cst04rtqelagr4.pdf` | 0.1 | `0a697a5f15c7b227` |
| `cst04rtqelagr6.pdf` | 0.1 | `28cd56c767b6678e` |
| `cst04rtqhssgr8.pdf` | 0.1 | `2b715998d1670ed4` |
| `cst04rtqmathgr3.pdf` | 0.1 | `87f084aad1e55603` |
| `cst04rtqmathgr4.pdf` | 0.1 | `389a3897ae414760` |
| `cst04rtqsciearth.pdf` | 0.1 | `6cd544fbf8e61e79` |
| `cstrtq13earthsci.pdf` | 0.6 | `8e2f45ec057d2a23` |
| `cstrtqalgebra.pdf` | 0.7 | `bab82b303bea0490` |
| `cstrtqalgebra2.pdf` | 0.6 | `ac5e6612cb432ae0` |
| `cstrtqbiology.pdf` | 0.7 | `9bfa0c398c91c999` |
| `cstrtqchemistry.pdf` | 0.5 | `a34b523754076196` |
| `cstrtqearthsci.pdf` | 0.6 | `31b9468b52f61084` |
| `cstrtqela10.pdf` | 2.7 | `488d8ab7f47ecf35` |
| `cstrtqela11.pdf` | 0.8 | `3cd43227618a3403` |
| `cstrtqela2.pdf` | 0.8 | `a3a18d3ad2b5c855` |
| `cstrtqela3.pdf` | 0.9 | `7d8fbbd090db9f6a` |
| `cstrtqela3nw.pdf` | 0.9 | `799a4005eb1e05a5` |
| `cstrtqela4.pdf` | 1.9 | `98b0a533983fa1c7` |
| `cstrtqela5.pdf` | 0.9 | `99acb4d3b8cbbc07` |
| `cstrtqela6.pdf` | 1.0 | `ac8ac9d0c73be237` |
| `cstrtqela7.pdf` | 0.9 | `69ef350827741b8e` |
| `cstrtqela8.pdf` | 1.0 | `bbedd14da6788ecf` |
| `cstrtqela9.pdf` | 1.6 | `5ecb71f5c1e3def6` |
| `cstrtqelagr11.pdf` | 0.8 | `543f1e4f196b1fc7` |
| `cstrtqelagr9.pdf` | 1.6 | `282a08ef9059267a` |
| `cstrtqgeomapr15.pdf` | 0.8 | `1baa2eb63846757c` |
| `cstrtqgeometry.pdf` | 0.7 | `78a71168faa1f1b9` |
| `cstrtqgr5elajul2012.pdf` | 0.9 | `52c4075dcc214260` |
| `cstrtqhss8.pdf` | 1.0 | `77a97b6515b0e88b` |
| `cstrtqhssmar18.pdf` | 0.4 | `215672f9ea1c5f7e` |
| `cstrtqhssworld.pdf` | 0.6 | `6940935c364ec28c` |
| `cstrtqmath2.pdf` | 4.1 | `d5fb25b9ceaa7872` |
| `cstrtqmath3.pdf` | 0.8 | `ac26ca0c34ab10f4` |
| `cstrtqmath4.pdf` | 0.8 | `7d461b531799d154` |
| `cstrtqmath5.pdf` | 0.6 | `0e1cc4ddc47cf678` |
| `cstrtqmath6.pdf` | 0.6 | `e2ba3c91d2295045` |
| `cstrtqmath7.pdf` | 0.7 | `a978516c0a8d98cd` |
| `cstrtqphysics.pdf` | 0.6 | `81d4e855ee44a2e7` |
| `cstrtqscience10.pdf` | 0.8 | `debc65a2d7edeb19` |
| `cstrtqscience5.pdf` | 2.3 | `140f52fc028b141a` |
| `cstrtqscience8.pdf` | 2.8 | `913bb9178f1c0217` |
| `cstrtqscigr8.pdf` | 2.7 | `2d4d4a5de32ff376` |
| `rtqalg1.pdf` | 0.4 | `e7ae7f73cc8a0814` |
| `rtqalg2.pdf` | 0.7 | `c955dbc38b603077` |
| `rtqbio.pdf` | 0.7 | `6f2ed4325ac920e5` |
| `rtqchem.pdf` | 0.3 | `a8690dd82000b10c` |
| `rtqearthscience.pdf` | 0.3 | `53eca53754115147` |
| `rtqgeom.pdf` | 0.6 | `1d34863563836130` |
| `rtqgr10ela.pdf` | 2.5 | `588f6efe9c9e3f42` |
| `rtqgr10history.pdf` | 0.4 | `5c55834822f30074` |
| `rtqgr10science.pdf` | 0.2 | `92f2d2b013005cd2` |
| `rtqgr11ela.pdf` | 0.7 | `fdb1e696f705b8f4` |
| `rtqgr11history.pdf` | 0.4 | `5e4261ccd4627ab9` |
| `rtqgr2ela.pdf` | 0.4 | `1df3ad10ecc71fc4` |
| `rtqgr2math.pdf` | 1.3 | `6d4682a1eb735513` |
| `rtqgr3ela.pdf` | 0.6 | `c2de64901f0ffdf4` |
| `rtqgr3math.pdf` | 0.7 | `5d59b56daaa58104` |
| `rtqgr4ela.pdf` | 1.1 | `2c84cf5492f4cd86` |
| `rtqgr4math.pdf` | 0.4 | `564ab6839cffcd50` |
| `rtqgr5ela.pdf` | 0.6 | `a935953cd4e2ef9a` |
| `rtqgr5math.pdf` | 0.4 | `5e3979242d4fb7fd` |
| `rtqgr5science.pdf` | 0.4 | `642305724c657906` |
| `rtqgr6ela.pdf` | 0.9 | `cf458cebc9d1d2b5` |
| `rtqgr6math.pdf` | 0.6 | `4d9951390ce466a9` |
| `rtqgr7ela.pdf` | 0.6 | `f47ddb9571219886` |
| `rtqgr7math.pdf` | 0.9 | `be4ca9160efe5a47` |
| `rtqgr8ela.pdf` | 0.7 | `03f2103f5beab1a4` |
| `rtqgr8history.pdf` | 0.5 | `ded01ffd27e2d81c` |
| `rtqgr8science.pdf` | 3.0 | `07a07ce092f91df4` |
| `rtqgr9ela.pdf` | 0.8 | `4d6a3b83089588aa` |
| `rtqgrworldhist.pdf` | 0.5 | `6d37133a606f3915` |
| `rtqphysics.pdf` | 0.3 | `2ae2a82b574d05f7` |

## Massachusetts MCAS (`ma`)

doe.mass.edu released items with answer keys

- **106 PDFs**, 170 MB, yielding **66 extracted items**
- rebuild: `python src/extract_ma.py`

| file | MB | sha256[:16] |
|---|---|---|
| `2019-g3-math.pdf` | 0.5 | `5c4d02a9f7bec3b5` |
| `2019-g4-math.pdf` | 2.1 | `66e03dd0b3c62731` |
| `2019-g5-math.pdf` | 0.4 | `5d00cc7b5a79979b` |
| `2019-g5-ste.pdf` | 2.6 | `781cc8db133cdd2a` |
| `2019-g6-math.pdf` | 0.4 | `958e00f17c1d30de` |
| `2019-g7-math.pdf` | 0.7 | `8a9079b439fbfdfc` |
| `2019-g8-math.pdf` | 0.5 | `47d6e681097c1d65` |
| `2019-g8-ste.pdf` | 0.4 | `10332ec341304c66` |
| `2021-g10-ela.pdf` | 0.6 | `4ba451a33dbb44a0` |
| `2021-g10-math.pdf` | 1.4 | `57e961881bc2a76f` |
| `2021-g3-ela.pdf` | 1.4 | `0b63788435921f7e` |
| `2021-g3-math.pdf` | 0.4 | `b88d4b517216060c` |
| `2021-g4-ela.pdf` | 2.3 | `febf1d5011d9e91f` |
| `2021-g4-math.pdf` | 0.5 | `2f3dbd71ff4cd0e5` |
| `2021-g5-ela.pdf` | 1.1 | `bb74262a35f19a23` |
| `2021-g5-math.pdf` | 0.4 | `73d4d13aaa5ade7d` |
| `2021-g5-ste.pdf` | 1.5 | `75492f24760b591f` |
| `2021-g6-ela.pdf` | 2.2 | `543c581c31329613` |
| `2021-g6-math.pdf` | 0.5 | `c6d0e6006630c87c` |
| `2021-g7-ela.pdf` | 1.1 | `ef7b1dd390e09f33` |
| `2021-g7-math.pdf` | 0.6 | `06603bfe70ee34d0` |
| `2021-g8-ela.pdf` | 0.9 | `a4c5d09eeee3f2cd` |
| `2021-g8-math.pdf` | 0.8 | `644b6caba5333793` |
| `2021-g8-ste.pdf` | 1.1 | `a43af18174440ffd` |
| `2022-g10-ela.pdf` | 2.1 | `e870446446d755e6` |
| `2022-g10-math.pdf` | 1.0 | `a9fdaf58742b8f9b` |
| `2022-g3-ela.pdf` | 1.9 | `d69ef62b9ec5ebef` |
| `2022-g3-math.pdf` | 0.7 | `b2499428036c63f8` |
| `2022-g4-ela.pdf` | 2.6 | `1959975eaf9b6739` |
| `2022-g4-math.pdf` | 0.9 | `66b39a0d0ace43cd` |
| `2022-g5-ela.pdf` | 3.5 | `bd8e59c3c999ca48` |
| `2022-g5-math.pdf` | 1.2 | `b3e5673841892b9a` |
| `2022-g5-ste.pdf` | 1.0 | `c3bca42eda4a7106` |
| `2022-g6-ela.pdf` | 1.8 | `4215247ab46640b1` |
| `2022-g6-math.pdf` | 1.5 | `156cca56e761a633` |
| `2022-g7-ela.pdf` | 0.6 | `8e125865f40a45c3` |
| `2022-g7-math.pdf` | 0.5 | `7d264c4420454220` |
| `2022-g8-ela.pdf` | 0.7 | `77731d434e7017ad` |
| `2022-g8-math.pdf` | 1.9 | `d6d87f2004ae9ef1` |
| `2022-g8-ste.pdf` | 0.9 | `93fccb735a5e0e54` |
| `2023-g10-ela.pdf` | 3.3 | `16ca1a69dcf52ab9` |
| `2023-g10-math.pdf` | 7.4 | `980c9276bc626170` |
| `2023-g3-ela.pdf` | 1.8 | `9e344b4f731334f6` |
| `2023-g3-math.pdf` | 0.5 | `abcfff970086c569` |
| `2023-g4-ela.pdf` | 1.4 | `f1d7e11d2a352484` |
| `2023-g4-math.pdf` | 0.7 | `b7f7299054d18d3f` |
| `2023-g5-ela.pdf` | 4.6 | `cd7d71fbecacbcfe` |
| `2023-g5-math.pdf` | 0.9 | `dec09eb751bb2d78` |
| `2023-g5-ste.pdf` | 14.5 | `c77052350db7a373` |
| `2023-g6-ela.pdf` | 0.6 | `97254258e6f2e3b1` |
| `2023-g6-math.pdf` | 1.0 | `810d10bc04d85f86` |
| `2023-g7-ela.pdf` | 3.4 | `69aca2b3cb1fc8f8` |
| `2023-g7-math.pdf` | 2.2 | `511f7b4ad14773c9` |
| `2023-g8-ela.pdf` | 1.1 | `6e9071723a13a0c6` |
| `2023-g8-math.pdf` | 1.6 | `fdcd33b47f1777b4` |
| `2023-g8-ste.pdf` | 1.5 | `9bd3b9b29d634f67` |
| `2024-g10-ela.pdf` | 2.2 | `b45bd8f053ed7ac4` |
| `2024-g10-math.pdf` | 1.5 | `30d541effdf04468` |
| `2024-g3-ela.pdf` | 0.3 | `32dbd5657d1709be` |
| `2024-g3-math.pdf` | 1.0 | `9ed0895e91445c05` |
| `2024-g4-ela.pdf` | 0.3 | `d8285a3c3b3a8849` |
| `2024-g4-math.pdf` | 1.2 | `2f04480be1216e47` |
| `2024-g5-ela.pdf` | 0.4 | `d0767c365ea53618` |
| `2024-g5-math.pdf` | 1.2 | `f1c2ffcf5fbf8661` |
| `2024-g5-ste.pdf` | 0.4 | `3438c113199dea3a` |
| `2024-g6-ela.pdf` | 0.5 | `8ea035a3b2ac1155` |
| `2024-g6-math.pdf` | 0.8 | `45ed262cafde5dc7` |
| `2024-g7-ela.pdf` | 0.9 | `fdc3cc1f59efa541` |
| `2024-g7-math.pdf` | 0.9 | `987370c29295449c` |
| `2024-g8-ela.pdf` | 0.4 | `08dfdb448efcf12d` |
| `2024-g8-math.pdf` | 0.9 | `05ff23b685a62461` |
| `2024-g8-ste.pdf` | 0.4 | `b69d37ab99b7629f` |
| `2025-g10-ela.pdf` | 1.3 | `94e82922ec001bb8` |
| `2025-g10-math.pdf` | 4.6 | `01b4fa2c071b3fa5` |
| `2025-g3-ela.pdf` | 0.9 | `3d8f5ac3c9b4c925` |
| `2025-g3-math.pdf` | 0.6 | `d675f6fc76a8d7c5` |
| `2025-g4-ela.pdf` | 1.0 | `62caf95b6bcebbdd` |
| `2025-g4-math.pdf` | 0.5 | `e6af1e153f0aef4e` |
| `2025-g5-ela.pdf` | 2.8 | `f44c5c5a78867f99` |
| `2025-g5-math.pdf` | 1.5 | `6dedef5ac359436a` |
| `2025-g5-ste.pdf` | 1.5 | `1a3c3010d59d3f99` |
| `2025-g6-ela.pdf` | 1.2 | `e73194c4d19a0f02` |
| `2025-g6-math.pdf` | 1.2 | `fcf2974638249a11` |
| `2025-g7-ela.pdf` | 2.0 | `c4df5e4d4c993383` |
| `2025-g7-math.pdf` | 1.5 | `ea7a1f2774b7f6f2` |
| `2025-g8-civics.pdf` | 2.5 | `15a84ae8c10935ab` |
| `2025-g8-ela.pdf` | 2.0 | `312f38c8a7ed645c` |
| `2025-g8-math.pdf` | 1.8 | `7dbfcaac5ce7cc01` |
| `2025-g8-ste.pdf` | 2.4 | `ebd28879c167a815` |
| `2026-g10-ela.pdf` | 1.9 | `5d5b9931b61a5be3` |
| `2026-g10-math.pdf` | 3.0 | `953c68821da891a5` |
| `2026-g3-ela.pdf` | 0.5 | `0831960a1c52d0da` |
| `2026-g3-math.pdf` | 2.6 | `d1d1ee5dfa684d29` |
| `2026-g4-ela.pdf` | 2.8 | `d82825b2350cdc75` |
| `2026-g4-math.pdf` | 2.4 | `6ba1de492050ddb7` |
| `2026-g5-ela.pdf` | 0.9 | `e8f991197e9ba0d3` |
| `2026-g5-math.pdf` | 1.9 | `c9006b2c15ad8082` |
| `2026-g5-ste.pdf` | 2.1 | `16e04612e459cc34` |
| `2026-g6-ela.pdf` | 0.8 | `4e9209b02388de19` |
| `2026-g6-math.pdf` | 2.1 | `e608c4f39580761e` |
| `2026-g7-ela.pdf` | 1.7 | `857fe3429c1a5571` |
| `2026-g7-math.pdf` | 2.2 | `c4597bfb402b6daf` |
| `2026-g8-civics.pdf` | 3.3 | `8378d797ca80b66a` |
| `2026-g8-ela.pdf` | 1.0 | `ae0776f457daa23f` |
| `2026-g8-math.pdf` | 2.5 | `3f40895abd0e6ed1` |
| `2026-g8-ste.pdf` | 2.3 | `4ca3d393e41bd9de` |

## New Jersey (`nj`)

NJSLA / PARCC / NJ ASK released items

- **35 PDFs**, 59 MB, yielding **32 extracted items**
- rebuild: `python src/extract_nj.py`

| file | MB | sha256[:16] |
|---|---|---|
| `gepa_math_scoring_guide.pdf` | 2.0 | `1b220a0de3f4e5d4` |
| `gepa_science_score_guide.pdf` | 2.3 | `0465abf52f91b8df` |
| `gepa_test_book_math.pdf` | 0.1 | `414e8fcad13ecc18` |
| `gepa_test_book_sci.pdf` | 4.1 | `87e65bc8121eafb5` |
| `gepa_test_book_ss.pdf` | 2.8 | `fcc74621fc152229` |
| `math_g3_key.pdf` | 0.2 | `83cecd75c9de3249` |
| `math_g3_test.pdf` | 1.0 | `2bd49976ca8a10b5` |
| `math_g4_key.pdf` | 0.1 | `099fd0fa43fd1a7b` |
| `math_g4_test.pdf` | 0.7 | `2082b62c951eb802` |
| `math_g5_key.pdf` | 0.2 | `3a177b2be0bb5317` |
| `math_g5_test.pdf` | 0.7 | `c2303dfefb66c2b5` |
| `math_g6_key.pdf` | 0.2 | `1f15afe63824a2c1` |
| `math_g6_test.pdf` | 0.9 | `429a63beb509f16b` |
| `math_g7_key.pdf` | 0.2 | `c6ed5ae0cf02d6ba` |
| `math_g7_test.pdf` | 1.1 | `9241c101892093d5` |
| `math_g8_key.pdf` | 0.2 | `2ca94fcda53e121a` |
| `math_g8_test.pdf` | 1.2 | `70589ea2a18f00f3` |
| `njask06_math.pdf` | 1.8 | `64563bbb35c04eda` |
| `njask06_sci_g4.pdf` | 2.8 | `8bcfa6d498d2e8a9` |
| `njask07_g5.pdf` | 6.9 | `1b374d217f6f6d38` |
| `njask07_g6.pdf` | 4.3 | `fa03784a79e39055` |
| `njask07_g7.pdf` | 4.0 | `4f1cddd94e7d04e8` |
| `njask07_key.pdf` | 2.7 | `4664e2778a0d7112` |
| `sci_g5_u1_key.pdf` | 0.5 | `4a937db1695a4839` |
| `sci_g5_u1_test.pdf` | 1.7 | `cdc6632aec9614ae` |
| `sci_g5_u2_key.pdf` | 0.8 | `531b54a3629dc8fe` |
| `sci_g5_u2_test.pdf` | 2.4 | `5e0c753af26d407e` |
| `sci_g5_u3_key.pdf` | 0.4 | `e1509cbd63059278` |
| `sci_g5_u3_test.pdf` | 2.8 | `46422c4db58e9152` |
| `sci_g8_u1_key.pdf` | 0.4 | `3fc2928944d3585f` |
| `sci_g8_u1_test.pdf` | 2.3 | `844fa0f1d186b3db` |
| `sci_g8_u2_key.pdf` | 0.6 | `b2ea28a6d4970d3b` |
| `sci_g8_u2_test.pdf` | 2.2 | `ba5aa1ea7fe69f00` |
| `sci_g8_u3_key.pdf` | 0.6 | `c69ff7e9ce38f2f3` |
| `sci_g8_u3_test.pdf` | 3.5 | `727457ca1ffddc8b` |
