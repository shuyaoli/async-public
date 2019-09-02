#include <stdio.h>

int main ()
{
  char parray[] = "cat";
  *parray = 'd';
  /* putchar(meow); */
  printf("%s\n", parray);
  return 0;
}
