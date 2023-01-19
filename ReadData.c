#include <stdio.h>
#include <stdlib.h>

unsigned char **readFile(){
   FILE *ptr;
   unsigned char *buffer = (unsigned char *)(malloc(sizeof(char) * (60000 * 784 + 16)));

   ptr = fopen("train-images-idx3-ubyte", "rb");
   fread(buffer, sizeof(char) * (60000 * 784 + 16), 1, ptr);
   buffer += 16;

   int index = 0;
   unsigned char **inputs = (unsigned char **)(malloc(sizeof(unsigned char *) * 60000));
   for (int i = 0; i < 60000; i++){
      inputs[i] = (unsigned char *)(malloc(sizeof(char) * 784));
      for (int j = 0 ; j < 784; j++){
         inputs[i][j] = buffer[index];
         index ++;
      }
   }

   buffer -= 16;
   free(buffer);
   fclose(ptr);
   return inputs;
}

int main(){
   
}
