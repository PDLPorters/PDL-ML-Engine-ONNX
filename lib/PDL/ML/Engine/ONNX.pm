package PDL::ML::Engine::ONNX;

use strict;
use warnings;
use Exporter 'import';
use Carp qw(croak);
use XSLoader;

use PDL::Core;
use PDL::Types;
use PDL::Slices;

our $VERSION = '0.011';

our @EXPORT_OK = qw(
    onnx_version
    available_providers
    load_session
);

XSLoader::load('PDL::ML::Engine::ONNX', $VERSION);

sub load_session {
    my ($path) = @_;
    croak 'load_session requires a model path' unless defined $path && length $path;
    return PDL::ML::Engine::ONNX::Session::_xs_create($path);
}

1;

__END__

=encoding UTF-8

=head1 NAME

PDL::ML::Engine::ONNX - ONNX Runtime C API bindings for PDL

=head1 SYNOPSIS

    use PDL;
    use PDL::ML::Engine::ONNX qw(onnx_version available_providers load_session);

    print 'ONNX Runtime: ', onnx_version(), "\n";
    print 'Providers: ', join(', ', available_providers()), "\n";

    my $sess = load_session('model.onnx');
    my @inputs  = $sess->input_names();
    my @outputs = $sess->output_names();

=head1 DESCRIPTION

C<PDL::ML::Engine::ONNX> binds L<PDL> to ONNX Runtime's pure-C API
(C<onnxruntime_c_api.h>) via C99 XS for cross-framework ML model inspection
and inference (PyTorch, TensorFlow, XGBoost, scikit-learn).

=head1 FUNCTIONS

=head2 onnx_version

=for ref

Return the linked ONNX Runtime version string.

=for usage

    $ver = onnx_version();

=head2 available_providers

=for ref

Return the list of available ONNX Runtime execution providers (e.g.,
C<CPUExecutionProvider>, C<CUDAExecutionProvider>).

=for usage

    @providers = available_providers();

=head2 load_session

=for ref

Load an C<.onnx> model file into a C<PDL::ML::Engine::ONNX::Session>.

=for usage

    $sess = load_session($path);

=head1 TUTORIAL

This tutorial adapts the ONNX Runtime C API quickstart to idiomatic Perl and
L<PDL>: querying the runtime environment, checking available execution
providers, and loading an C<.onnx> model session to inspect its input and
output tensor names.

=head2 1. Inspect the ONNX Runtime Environment

    use strict;
    use warnings;
    use PDL::ML::Engine::ONNX qw(onnx_version available_providers load_session);

    print 'ONNX Runtime version: ', onnx_version(), "\n";

    my @providers = available_providers();
    print 'Available providers:  ', join(', ', @providers), "\n";

    if (grep { $_ eq 'CUDAExecutionProvider' } @providers) {
        print 'CUDA acceleration is available', "\n";
    }

=head2 2. Load a Model Session and Inspect Graph Signatures

Use C<load_session> to open any exported C<.onnx> model (from PyTorch,
TensorFlow, scikit-learn, or XGBoost) and query its input and output tensor
names:

    my $model_path = 'model.onnx';
    my $sess       = load_session($model_path);

    my @inputs  = $sess->input_names();
    my @outputs = $sess->output_names();

    print 'Inputs:  ', join(', ', @inputs), "\n";
    print 'Outputs: ', join(', ', @outputs), "\n";

The underlying C<OrtSession> handle is released automatically when C<$sess>
goes out of scope.

See F<examples/quickstart.pl> in the distribution for a complete runnable
script.

=head1 TYPES

Supports standard real numeric PDL types (C<byte>, C<short>, C<long>, C<longlong>, C<float>, C<double> as supported by the underlying library). Complex types (C<cfloat>, C<cdouble>) are not supported by this backend.

=head1 SEE ALSO

L<PDL>, L<PDL::Torch>, L<PDL::XGBoost>, L<PDL::FAISS>, L<PDL::GGML>

=head1 AUTHOR

C.J. Collier <cjac@google.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2026 Google LLC

This library is free software; you can redistribute it and/or modify it under
the same terms as Perl itself.

=cut
