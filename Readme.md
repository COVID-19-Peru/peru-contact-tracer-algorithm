Peru contact tracer algorithm
========================

En el presente proyecto se propone la implementación un modelo de trazado de contacto digital epidemiológico para estimar la probabilidad de infección de los ciudadanos en Perú con el COVID-19. Esta propuesta deberá complementar la aplicación móvil **Perú en tus manos**[1] concebida y promovida por este equipo de científicos y desarrollada por la sociedad civil, universidades, gobierno y empresas privadas. En detalle, el modelo que proponemos se nutrirá de las ubicaciones obtenidas del GPS y encuentros de otros dispositivos cercanos a través de Bluetooh del dispositivo de las personas usando el App antes mencionada y estimará la probabilidad de infección de dichos usuarios.



----------


Archivos
-------------

El presente repositorio consta de tres archivos. 

#### <i class="icon-file"></i> Script principal

> **covid19.py:**

> -Es el script que contiene una implemetación del trazado de contactos digital epidemiologico.

#### <i class="icon-file"></i> Archivo de configuración

> **config.cfg:**

> -Es el archive de configuración que consta de dos partes. 
> 1) La primera parte **[path]** donde:
> * *outputFilePath* especifica la carpeta donde los eventos (encuentros entre infectados y no infectados) se almacenan. 
> * *matrix* es el path al archivo numpy que contienen la matriz de probabilidades
> * *credential* es el path al archivo json donde esta la credencial para acceder a la base de datos
>  2) La segunda parte **[parameters]**
>  * *t_start* Fecha y hora de inicio del estudio
>  * *t_end* Fecha y hora de fin del estudio
>  * *distance* distancia en metros para considerar el filtro espacial de encuentros
>  * *interval* tiempo en minutos para el filtro temporal de los encuentros

#### <i class="icon-file"></i> Matriz de probabilidades

> **matrizProba.npy:**

> -Archivo numpy que contienen la matriz de probabilidad de ontagio en función de la distancia.

#### <i class="icon-hdd"></i> Ejecutar el script
```
python covid19.py config.cfg
```


----------


  [1]: El equipo conformado por  Lucia Del Carpio del INSEAD Paris;  Gianmarco Leóon-Ciliotta Universitat Pompeu Fabra & Barcelona GSE & IPEG & CEPR, España; Kristian López Vargas; Gonzalo Panizo Universidad Nacional de Ingeniería; Hugo Alatrista-Salas, Miguel Nunez-del-Prado, Alvaro Talavera de la Universidad del Pacífico   fue el que propuso inicialmente realizar el trazado de contactos y diseñó la estrategia que se viene implementando. Pero somos un equipo grande de profesionales (del gobierno, academia y sector privado) trabajando en el ecosistema del aplicativo. Además del equipo académico impulsor, está el equipo de desarrollo en el que se encuentran las empresas Tekton Labs, Kambista, Sapia, Sr. Burns, Media Labs y el Grupo Alicorp. Asimismo, se nos unió un equipo de la Universidad de Tecnología e Ingeniería para colaborar con el desarrollo de los algoritmos. En ese equipo 

