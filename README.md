# Enhancing Disease and Pest Detection in Greenhouse Tomato Cultivation Using Advanced Machine Learning on New Dataset of Images

Código organizado a partir dos notebooks disponibilizados por Grasielli Barreto Zimmermann para o trabalho de Zimmermann, Pellenz, Costa e Britto Jr., *Journal of the Brazilian Computer Society* 31(1), 187–202 (2025). [DOI: 10.5753/jbcs.2025.4581](https://doi.org/10.5753/jbcs.2025.4581). [Datasets TLID e PTLID](https://data.mendeley.com/datasets/kt64b2kh89/1).

## Organização

| Pasta | Entrada | Imagens usadas no treino | Modelos nos textos recebidos |
| --- | --- | --- | --- |
| `TLID/` | `python -m TLID.train` | folhas redimensionadas para 224 × 224 | CNN, VGG19, ResNet50 |
| `PTLID/` | `python -m PTLID.train` | patches 32 × 32 | MobileNet, VGG19, ResNet50 |

Cada pasta tem seus próprios arquivos `data.py`, `models.py`, `train.py`, `evaluate.py` e `ensemble.py`. Os datasets e os pesos dos modelos não estão incluídos.

## Estrutura esperada dos dados

```text
dataset_tlid/Tomato/0-Saudaveis/imagem001.jpg
dataset_tlid/Tomato/1-Mosca Minadora/imagem002.jpg
dataset_ptlid/Tomato/0-Patch_Healthy/patch001.jpg
dataset_ptlid/Tomato/1-Patch_Miner/patch002.jpg
```

O programa descobre os nomes reais das classes nas pastas. Não renomeie uma classe entre treino e avaliação. Para projetos com outras pastas de planta, mantenha a mesma forma `raiz/planta/classe/imagem`.

## Executar

Use Python e um ambiente com TensorFlow, preferencialmente GPU para a TLID:

```bash
python -m pip install -r requirements.txt
python -m TLID.train --data dataset_tlid --output outputs_tlid
python -m TLID.evaluate --data dataset_tlid --output outputs_tlid --mode hard
python -m PTLID.train --data dataset_ptlid --output outputs_ptlid
python -m PTLID.evaluate --data dataset_ptlid --output outputs_ptlid --mode hard
```

Para treinar só um modelo, acrescente `--models cnn` (TLID) ou `--models mobilenet` (PTLID). Para votação soft, use `--mode soft --weights 1 1 1` na avaliação. Pesos ajustados devem vir de dados de validação, nunca do teste. O pré-processamento específico de cada modelo é aplicado antes do treino e da avaliação e não está embutido no arquivo `.keras`.

## Estado de reprodução

Este código é uma organização dos dois textos exportados do Colab, com correções para execução em arquivos Python. Ainda **não foi executado com os datasets** neste repositório e não deve ser apresentado como reprodução exata das métricas publicadas. O artigo descreve um trio CNN/VGG19/ResNet50 e sete classes; o texto PTLID recebido contém MobileNet/VGG19/ResNet50 e oito classes, incluindo Background. O texto TLID mostra 15.271 imagens carregadas, enquanto o artigo informa 15.256. O código organizado também usa divisões estratificadas, pré-processamento específico de cada arquitetura e `sparse_categorical_crossentropy`, que diferem de trechos dos notebooks. O número de imagens, as classes, o protocolo de separação por folha/planta e as métricas precisam ser conferidos com os dados originais antes de uma alegação de reprodução.
