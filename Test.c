#include <stdio.h>
#include <stdlib.h>

int main(){
     int* x[5];
     
     for (int i = 0; i < 5; i++){
        x[i] = malloc(sizeof(int *) * 5);
     }

     for (int i = 0; i < (sizeof(x)/sizeof(x[0])); i++){
        for (int j = 0; j < 5; j++){
            x[i][j] = j;
        }
     }

     for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j ++){
            printf("%d", x[i][j]);
        }
        printf("\n");
     }
}

