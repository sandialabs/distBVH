/*
 * distBVH 1.0
 *
 * Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <bvh/kdop.hpp>
#include <bvh/vis/vis_bvh.hpp>
#include <bvh/tree_build.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include <bvh/math/vec.hpp>
#include <bvh/tree.hpp>
#include <vector>
#include <random>

#include <vtkPolyData.h>
#include <vtkAppendPolyData.h>
#include <vtkSphereSource.h>

class pelement
{
public:
  
  explicit pelement( std::size_t _index, const bvh::m::vec3d &_p )
    : m_index( _index ), m_point( _p ) {}
  
  const bvh::m::vec3d &centroid() const { return m_point; }
  
  bvh::dop_26d kdop() const { return bvh::dop_26d::from_sphere( m_point, 0.01 ); }
  
  std::size_t global_id() const { return m_index; }
  
  const bvh::m::vec3d &point() const { return m_point; }
  
private:
  
  const std::size_t m_index;
  bvh::m::vec3d m_point;
};

decltype(auto) get_entity_global_id( const pelement &_element )
{
  return _element.global_id();
}

decltype(auto) get_entity_kdop( const pelement &_element )
{
  return _element.kdop();
}

decltype(auto) get_entity_centroid( const pelement &_element )
{
  return _element.centroid();
}

int
main( int _argc, char **_argv )
{
  if ( _argc != 2 )
  {
    std::cerr << "Usage: " << _argv[0] << " <output prefix>";
    return 1;
  }
  
  auto elements = std::vector< pelement >{};
  elements.reserve( 100 );
  
  std::mt19937 gen( 0 );
  std::uniform_real_distribution<> locgen( 0.0, 1.0 );
  for ( std::size_t i = 0; i < 100; ++i )
  {
    bvh::m::vec3d p( locgen( gen ), locgen( gen ), locgen( gen ) );
    
    elements.emplace_back( i, p );
  }

  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  std::ostringstream fname;
  fname << _argv[1] << ".vtp";
  auto out = std::ofstream( fname.str().c_str() );
  bvh::vis::write_bvh( out, tree );
  out.close();
  
  for ( std::size_t i = 0; i < tree.depth(); ++i )
  {
    std::ostringstream fname;
    fname << _argv[1] << "_" << i << ".vtp";
    auto out = std::ofstream( fname.str().c_str() );
    bvh::vis::write_bvh_level( out, tree, i );
  }
  
  // Output sphere elements
  auto poly_list = bvh::vis::make_vtkobj< vtkAppendPolyData >();
  for ( auto &&el : elements )
  {
    auto sphere = bvh::vis::make_vtkobj< vtkSphereSource >();
    sphere->SetRadius( 0.01 );
    sphere->SetCenter( el.point().x(), el.point().y(), el.point().z() );
    sphere->SetThetaResolution( 32 );
    sphere->SetPhiResolution( 16 );
    sphere->Update();
    
    poly_list->AddInputData( sphere->GetOutput() );
    
    sphere.release();
  }
  
  poly_list->Update();
  
  auto writer = bvh::vis::make_vtkobj< vtkXMLPolyDataWriter >();
  writer->SetInputData( poly_list->GetOutput() );
  writer->WriteToOutputStringOn();
  writer->Write();
  
  std::stringstream efname;
  efname << _argv[1] << "_elements.vtp";
  
  auto elements_out = std::ofstream( efname.str().c_str() );
  
  elements_out << writer->GetOutputString();
}
