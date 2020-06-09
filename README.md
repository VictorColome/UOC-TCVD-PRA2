# UOC-TCVD-PRA2

## Título
Práctica 2 de Tipología y ciclo de vida de los datos en la UOC por los alumnos Carlos Marcos y Víctor Colomé

### Descripción
En esta práctica se ha decidido por parte de los alumnos utilizar Jupyter Notebook, ya que esta herramienta permite una sencilla integración de código python y comentarios de texto enriquecido, permitiendo comentar fácilmente los comandos utilizados. Por tanto, la entrega final de la práctica se compone del fichero ipynb, que contiene el código y comentarios. Adicionalmente y con el objetivo de facilitar la corrección por parte del profesor, se entrega el HTML generado a partir del mismo.

Dado que Git no permite la integración sencilla de ficheros ipynb (como sí permite de .py, por ejemplo), para el desarrollo se ha utilizado un fichero ipynb que contenía llamadas a las funciones python desarrolladas en otro fichero script.py. Este último es el que ha permitido el desarrollo en paralelo de ambos alumnos. Una vez las funciones estaban desarrolladas y probadas, se han integrado en el fichero ipynb junto con las explicaciones pertinentes. En el proyecto GitHub se han mantenido estos ficheros para que el profesorado pueda verificar el trabajo realizado si es necesario.

### Estructura del código
Los ficheros y carpetas más relevantes de esta práctica son los siguientes:
* victor_carlos_PRA2.ipynb. Fichero principal de la práctica, que contiene tanto el código Python como las explicaciones sobre los gráficos y datos.
* victor_carlos_PRA2.html. Fichero resultado del anterior que puede consultarse directamente en un navegador sin ejecutar el código Python y que contiene dichas explicaciones y gráficos
* script.py. Fichero que se ha utilizado durante el desarrollo para el trabajo en paralelo de los alumnos usando GitHub
* requirements.txt. Fichero con los requisitos del proyecto
* README.md. Este fichero
* sample. Directorio del IDE utilizado
	* 	__init__.py. Fichero de inicialización del proyecto del IDE
* doc. Directorio con la documentación del proyecto:
	* Entrega.txt. Fichero que se entrega en el aula
	* enunciado.pdf. Fichero con el enunciado de la práctica
	* Guión PAC2 Tipología de Datos.pdf. Fichero de entrega tal y como especifican las instrucciones del enunciado
* data. Directorio con los ficheros de datos utilizados
	* adult.data. Fichero con el juego de datos
	* adult.names. Fichero con los nombres de las columnas del juego de datos
	* adult_processed.data. Fichero con los datos resultado del procesamiento

