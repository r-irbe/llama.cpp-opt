#include "server-app.h"
#include "server-context.h"

int main(int argc, char ** argv) {
    server_context context;
    return server_run(argc, argv, context);
}
