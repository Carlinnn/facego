# FaceGo

Sistema de reconhecimento facial desenvolvido em Go, com processamento 100% local — sem APIs externas, sem custos por requisição.

![Go](https://img.shields.io/badge/Go-1.27+-00ADD8?style=flat&logo=go)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.18-FF6F00?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## Sobre

FaceGo responde três perguntas:

1. **Existe um rosto nessa imagem?**
2. **Esse rosto pertence a alguma pessoa cadastrada?**
3. **Se sim, quem é essa pessoa?**

Tudo processado localmente — nenhum dado biométrico sai da sua máquina.

---

## Como funciona

```
Foto / Câmera
      │
      ▼
  SCRFD (detecção)
      │
   Rosto?
   /     \
 NÃO     SIM
  │       │
Erro   Alignment
          │
       ArcFace (embedding 512-d)
          │
    Cosine Similarity
    contra embeddings
    cadastrados no SQLite
          │
     ┌────┴────┐
  Acima      Abaixo
 do limiar  do limiar
     │           │
IDENTIFICADO  DESCONHECIDO
```

### Modelos utilizados

| Modelo | Tarefa | Tamanho |
|--------|--------|---------|
| SCRFD `det_10g.onnx` | Detecção facial + landmarks 5 pontos | 16 MB |
| ArcFace `w600k_r50.onnx` | Embedding facial 512-d | 166 MB |

Ambos do pacote [buffalo_l do InsightFace](https://github.com/deepinsight/insightface).

---

## Stack

- **Backend:** Go + Gin
- **Banco de dados:** SQLite (`modernc.org/sqlite` — pure Go, sem CGO)
- **Inferência:** ONNX Runtime 1.18 via `onnxruntime_go`
- **Frontend:** HTML5 + CSS3 + JavaScript Vanilla
- **Webcam:** `navigator.mediaDevices.getUserMedia()`

---

## Requisitos

- Go 1.27+
- GCC (MinGW-w64 no Windows, para CGO do binding ONNX)
- [ONNX Runtime 1.18.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1) — DLL/SO no diretório do binário

---

## Instalação

### 1. Clone

```bash
git clone https://github.com/Carlinnn/facego.git
cd facego
```

### 2. Dependências Go

```bash
go mod download
```

### 3. ONNX Runtime

Baixe o pacote para seu sistema operacional em:
https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1

Coloque `onnxruntime.dll` (Windows) ou `libonnxruntime.so` (Linux) no mesmo diretório do binário compilado.

### 4. Modelos

Baixe o pacote `buffalo_l` do InsightFace:
https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

Extraia e copie para `models/`:

```
models/
├── scrfd.onnx      ← det_10g.onnx renomeado
└── arcface.onnx    ← w600k_r50.onnx renomeado
```

### 5. Build e execução

```bash
go build -o facego ./cmd/facego
./facego
```

O servidor sobe na porta `8080` por padrão.

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `PORT` | `8080` | Porta HTTP |
| `DATABASE_PATH` | `./data/facego.db` | Caminho do banco SQLite |
| `FACES_DIR` | `./data/faces` | Diretório de fotos cadastradas |
| `MODELS_DIR` | `./models` | Diretório dos modelos ONNX |
| `THRESHOLD` | `0.5` | Limiar de similaridade para identificação |
| `MAX_IMAGE_SIZE` | `10485760` | Tamanho máximo de imagem (bytes) |

---

## API

### Health check

```http
GET /api/v1/health
```
```json
{ "status": "ok" }
```

### Cadastrar pessoa

```http
POST /api/v1/persons
Content-Type: multipart/form-data

name=Carlos Tavares
image=<arquivo>
```
```json
{
  "success": true,
  "person": { "id": 1, "uuid": "019a82d4-...", "name": "Carlos Tavares" }
}
```

### Listar pessoas

```http
GET /api/v1/persons
```

### Excluir pessoa

```http
DELETE /api/v1/persons/:id
```

Remove cadastro, embedding e foto armazenada.

### Detectar rosto

```http
POST /api/v1/detect
Content-Type: multipart/form-data

image=<arquivo>
```
```json
{ "detected": true, "faces": 1, "confidence": 0.998 }
```

### Identificar

```http
POST /api/v1/identify
Content-Type: multipart/form-data

image=<arquivo>
```

**Identificado:**
```json
{
  "identified": true,
  "person": { "id": 1, "uuid": "...", "name": "Carlos Tavares", "photo": "/faces/.../face.jpg" },
  "similarity": 0.91
}
```

**Desconhecido:**
```json
{ "identified": false, "person": null, "message": "Rosto não identificado." }
```

---

## Interface Web

| Rota | Descrição |
|------|-----------|
| `/` | Página inicial |
| `/register` | Cadastro via câmera ou upload |
| `/identify` | Identificação via câmera ou upload |
| `/persons` | Lista de pessoas cadastradas |

---

## Estrutura do projeto

```
facego/
├── cmd/facego/         # Ponto de entrada
├── internal/
│   ├── api/            # Handlers e rotas Gin
│   ├── config/         # Configuração via env
│   ├── database/       # SQLite + migrations
│   ├── face/           # SCRFD, ArcFace, similaridade
│   └── person/         # Model, repositório, serviço
├── models/             # Modelos ONNX (não versionados)
├── web/                # Frontend HTML/CSS/JS
│   ├── css/
│   └── js/
├── tools/inspect_onnx/ # Utilitário para inspecionar modelos
├── data/               # Banco e fotos (não versionados)
└── go.mod
```

---

## Armazenamento

Embeddings são armazenados como `BLOB` no SQLite. Na inicialização, todos são carregados em memória para comparação rápida — suficiente para milhares de pessoas.

Estrutura de fotos:

```
data/faces/
└── {uuid}/
    └── face.jpg
```

---

## Segurança e privacidade

- Nenhum dado biométrico é transmitido externamente
- Imagens limitadas a 10 MB por padrão
- Apenas JPEG e PNG aceitos
- Nomes de arquivo gerados internamente (UUID)
- Exclusão completa: cadastro + embedding + foto
- `data/` nunca é versionado

---

## Melhorias futuras

- Múltiplos embeddings por pessoa
- Reconhecimento em tempo real (vídeo contínuo)
- Liveness detection / anti-spoofing
- Autenticação e controle de acesso
- Suporte a GPU
- Busca vetorial (FAISS) para escala

---

## Licença

MIT
