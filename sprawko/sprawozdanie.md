# Programowanie Równoległe - CUDA

Wersja pierwsza

### Autorzy

**Grupa dziekańska:** 4  
**Grupa labolatoryjna:** 7  
**Termin zajęć:** czwartek, 16:50


Tymoteusz Jagła 151811 - tymoteusz.jagla@student.put.poznan.pl  
Kaper Magnuszewski 151746 - kacper.magnuszewski@student.put.poznan.pl

### Sprawozdanie

_Wymagany termin oddania sprawozdania -_ 31.05.2024  
_Rzeczywisty termin oddania sprawozdania -_ 31.05.2024  

## Cel zadania
Celem ćwiczenia jest praktyczne zapoznanie z zasadami programowania równoległego procesorów kart graficznych (PKG), zapoznanie z zasadami optymalizacji kodu dla PKG oraz ocena prędkości przetwarzania przy użyciu PKG i poznanie czynników warunkujących realizację efektywnego przetwarzania.

## Opis zadania

Zadanie polega na sumowaniu wartości tablicy o wielkości $N$ x $N$, które znajdują się w określonym przez „promień” $R$ obszarze. Tablica wynikowa ma rozmiar $(N-2R)^2$, sumy obliczane są dla wszystkich pozycji w odległości większej lub równej $R$ od krawędzi tablicy wejściowej.

Tablica Dwuwymiarowa $TAB[N][N]$ (o wierszu długości $N$, słowo tablicy $TAB[i][j]$ jest dostępne jako $TAB[i*N+j]$). Dla tablicy wejściowej $TAB$ należy wyliczyć tablicę wyjściową $OUT[N-2R][N-2R]$ (gdzie $N>2R$)
zawierającą sumy elementów w „promieniu” $R$. Każdy element tablicy wyjściowej to suma $(2*R+1)*(2*R+1)$ wartości.

Przykładowo dla $R=1$ $OUT[i][j]=TAB[i][j]+ TAB[i][j-1]+ TAB[i][j+1]+ TAB[i-1][j-1]+ TAB[i-1][j]+TAB[i-1][j+1]+ TAB[i+1][j]+ TAB[i+1][j-1]+ TAB[i+1][j+1]$.

## Wykorzystany system obliczeniowy

### Procesor (CPU)

- Model: 13th Gen Intel® Core(TM) i5-13600KF
- Liczba procesorów fizycznych: 14
  - 6 Performance-cores
  - 8 Efficient-cores
- Liczba procesorów logicznych: 20
  - 2 wątki na pojedyńczy Performance-core
  - 1 wątek na pojedyńczy Effitient-core
- Oznaczenie typu procesora: KF
- Taktowanie procesora:
  - Minimalne: 800MHz
  - Maksymalne: 51000MHz
- Wielkości pamięci podręcznej procesora: 
  - L1d cache: 544 KiB (14 instancji)
  - L1i cache: 704 KiB (14 instancji)
  - L2 cache: 20 MiB (8 instancji)
  - L3 cache: 24 MiB (1 instancja)
- Organizacja pamięci podręcznej: Intel® Smart Cache

### Jednostka przetwarzania graficznego (GPU)
 
 - Model: NVIDIA GeForce RTX 4070 SUPER 12G VENTUS 2X OC 
 - Nazwa technologii: Ada Lovelace
 - Producent: Micro-Star International
 - Układ graficzny: AD104-350
 - Parametr CUDA compute capability: 8.9
 - Liczba tranzystorów: 35.800 milionów
 - Proces technologiczny: 5nm
 - Rdzenie CUDA: 7168
 - Jednostki TMU: 224
 - Jednostki ROP: 80
 - Jednostki RT: 56
 - Jednostki Tensor: 224
 - Pamięć VRAM: 12 GB GDDR6X
 - Magistrala pamięci: 192-bitowa
 - Taktowanie pamięci: 1313 MHz
 - Taktowanie pamięci efektywne: 21000 MHz
 - Przepustowość pamięci: 504 GB/s
 - Taktowanie rdzenia (bazowe): 1980 MHz
 - Taktowanie rdzenia (boost): 2505 MHz
 - Pamięć cache L2: 48 MB
 - Pobór mocy (TGP): 220 W
 - Wersja sterownika: NVIDIA 551.61

### System Operacyjny

- Nazwa systemu operacyjnego: Microsoft Windows 11 N
- Oprogramowanie wykorzystane do przygotowania kodu wynikowego: Visual Studio 2022
- Oprogramowanie wykorzystane do przeprowadzenia testów: NVIDIA Nsight Compute 2024.02

## Wersje programów

### Wykorzystane zmienne:
 - $N$ - wielkość wymiaru tablicy wejściowej
 - $R$ - promień, w jakim realizowane jest sumowanie
 - $K$ - liczba wyników obliczanych przez jeden wątek
 - $BS$ - wielkość wymiaru bloku wątków
 - $tab[N * N]$ - tablica wejściowa o wielkości $N*N$
 - $out[(N-2R)*(N-2R)]$ - tablica wyjściowa o wielkości $(N-2R)*(N-2R)$


### Algorytm rozwiązujący problem sekwencyjnie dla głównego procesora komputera
Poniższa funkcja to część programu, która służy obliczeniom sekwencyjnym (wykorzystuje jeden wątek) przy użyciu głównego procesora komputera. 

  ***Kod 1. Obliczenia sekwencyjne***
~~~ { #K1 .cpp .numberLines caption="Kod 1. Obliczenia sekwencyjne"}

void sequential(float tab[N*N], float out[(N-2*R)*(N-2*R)])
{
	for (int i = R; i < N - R; i++) {
		for (int j = R; j < N - R; j++) {
			float sum = 0;
			for (int x = i - R; x <= i + R; x++) {
				for (int y = j - R; y <= j + R; y++) {
					sum += tab[x * N + y];
				}
			}
			out[(i - R) * (N - 2 * R) + j - R] = sum;
		}
	}
}
~~~


### Algorytm rozwiązujący problem równolegle wykorzystujący pamięć globalną
Poniższy kernel służy obliczeniom równoległym przy użyciu technologii CUDA procesora graficznego. Algorytm efektywnie wykorzystuje dane w pamięci globalnej karty - wątki realizują jednocześnie dostępy do sąsiednich elementów w pamięci globalnej.

  ***Kod 2. Obliczenia przy użyciu CUDA***
~~~ { #K2 .cpp .numberLines caption="Kod 2. Obliczenia CUDA"}
__global__ void localKernel(float* tab, float* out, int* kkk)
{
    int i = (threadIdx.x + blockIdx.x * blockDim.x) * *kkk;
    int j = (threadIdx.y + blockIdx.y * blockDim.y);

    for (int k = 0; k < *kkk; k++) {
        int ik = i + k;
        if (ik < OUTSIZE) {
			float sum = 0;
            for (int y = 0; y <= 2*R; y++) {
                int jy = (j + y)*N;
				for (int x = 0; x <= 2*R; x++) {
					sum += tab[jy + (ik + x)];
				}
            }
            out[(j) * (OUTSIZE) + ik] = sum;
        }
    }
}
~~~

### Wywołanie procedury kernela


  ***Kod 3. Wywołanie kernela***
~~~ { #K3 .cpp .numberLines caption="Kod 2. Wywołanie kernela"}
cudaError_t sumLocalWithCuda(float* tab, float* out) 
{
    float* dev_tab = 0;
    float* dev_out = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_tab, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, (OUTSIZE) * (OUTSIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_tab, tab, N*N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadsMatrix(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksMatrix(ceil((OUTSIZE) / (float)BLOCK_SIZE / K), ceil((OUTSIZE) / (float)BLOCK_SIZE));

    localKernel<<< blocksMatrix, threadsMatrix >>>(dev_tab, dev_out);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "local launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(out, dev_out, (OUTSIZE) * (OUTSIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_tab);
    cudaFree(dev_out);
    
    return cudaStatus;
}
~~~

## Wnioski

### Generowanie wartości testowych i sprawdzanie poprawności obliczeń
Do generowania wartości testowych użyliśmy liczb rzeczywistych pseudolosowych generowanych za pomocą funkcji `rand()` z biblioteki standardowej. Aby przetestować poprawność obliczeń przy użyciu algorytmu rozwiązującego problem równolegle porównywaliśmy za każdym razem jego wyjście do wyjścia funkcji obliczającej wartości tablicy sekwencyjnie. Ostateczna wersja algorytmu równoległego była w stanie rozwiązać zadany problem obliczeniowy poprawnie przy każdej próbie.
