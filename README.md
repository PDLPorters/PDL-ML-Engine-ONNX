# PDL::ML::Engine::ONNX

ONNX Runtime C API (`onnxruntime_c_api.h`) bindings for [PDL](https://pdl.perl.org/), enabling cross-framework ML model inspection and session management from Perl.

## Prerequisites

- Perl 5.10+
- `PDL` (>= 2.080)
- ONNX Runtime C library and headers (`onnxruntime_c_api.h`)

## Installation

```bash
perl Makefile.PL
make
make test
make install
```

Or with `cpanm`:

```bash
cpanm .
```

## Quickstart

```perl
use strict;
use warnings;
use PDL::ML::Engine::ONNX qw(onnx_version available_providers load_session);

print 'ONNX Runtime: ', onnx_version(), "\n";
print 'Providers: ', join(', ', available_providers()), "\n";

my $sess    = load_session('model.onnx');
my @inputs  = $sess->input_names();
my @outputs = $sess->output_names();
```

## Examples & Documentation

- Runnable quickstart: [examples/quickstart.pl](examples/quickstart.pl) (`perl examples/quickstart.pl [model.onnx]`)
- Full API documentation and tutorial: `perldoc PDL::ML::Engine::ONNX`
