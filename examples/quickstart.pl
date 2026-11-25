# Copyright 2026 Google LLC
#
# Licensed under the the same terms as Perl itself (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

use strict;
use warnings;
use PDL::ML::Engine::ONNX qw(onnx_version available_providers load_session);

# 1. Inspect linked ONNX Runtime version and execution providers
print 'ONNX Runtime version: ', onnx_version(), "\n";

my @providers = available_providers();
print 'Available providers:  ', join(', ', @providers), "\n";

# 2. Load an ONNX model if a path is supplied on the command line
my ($model_path) = @ARGV;
if (defined $model_path) {
    my $sess    = load_session($model_path);
    my @inputs  = $sess->input_names();
    my @outputs = $sess->output_names();

    print 'Loaded model: ', $model_path, "\n";
    print 'Inputs:  ', join(', ', @inputs), "\n";
    print 'Outputs: ', join(', ', @outputs), "\n";
} else {
    print 'Pass a path to a .onnx file to inspect its input/output tensors:', "\n";
    print '  perl examples/quickstart.pl path/to/model.onnx', "\n";
}
