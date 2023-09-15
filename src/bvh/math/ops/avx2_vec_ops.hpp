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
#ifndef INC_BVH_MATH_OPS_AVX2_VEC_OPS_HPP
#define INC_BVH_MATH_OPS_AVX2_VEC_OPS_HPP

#include "../vec.hpp"

#include "../storage/vec_storage.hpp"
#include "vec_base_ops.hpp"

#ifdef BVH_SIMD

namespace bvh
{
  namespace m
  {
    namespace ops
    {
      namespace detail
      {
        inline __m256i
        make_imask64( unsigned _mask )
        {
          return _mm256_setr_epi64x( -1LL * ( _mask & 0x1 ), -1LL * ( ( _mask >> 1 ) & 0x1 ),
                                     -1LL * ( ( _mask >> 2 ) & 0x1 ), -1LL * ( ( _mask >> 3 ) & 0x1 ) );
        }

        inline __m256d
        make_mask( unsigned _mask )
        {
          return _mm256_castsi256_pd( _mm256_setr_epi64x( -1LL * ( _mask & 0x1 ), -1LL * ( ( _mask >> 1 ) & 0x1 ),
                                                          -1LL * ( ( _mask >> 2 ) & 0x1 ),
                                                          -1LL * ( ( _mask >> 3 ) & 0x1 ) ) );
        }

        template< std::size_t... Indices >
        struct storage_set_op_impl< vec, double, std::index_sequence< Indices... > >
        {
          using result_type = storage_type< double, sizeof...( Indices ) >;

          template< std::size_t TIX, std::size_t TIY = 0, std::size_t TIZ = 0, std::size_t TIW = 0 >
          static auto
          assign( result_type &_res, double _x, double _y = 0.0, double _z = 0.0, double _w = 0.0 )
          {
            _res.chunks[TIX / 4] = _mm256_setr_pd( _x, _y, _z, _w );
          }

          template< std::size_t TIX, std::size_t TIY, std::size_t TIZ, std::size_t TIW, std::size_t... TIndices >
          static auto
          assign( result_type &_res, double _x, double _y, double _z, double _w, expand< double, TIndices >... _args )
          {
            _res.chunks[TIX / 4] = _mm256_setr_pd( _x, _y, _z, _w );
            assign< TIndices... >( _res, _args... );
          }

          static auto
          execute( expand< double, Indices >... _args )
          {
            result_type ret;
            assign< Indices... >( ret, _args... );

            return ret;
          }
        };
      } // namespace detail

      template< unsigned N >
      struct init_undefined_storage_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute()
        {
          storage_type< double, N > ret{};

          // Basically a no-op
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.chunks[i] = _mm256_undefined_pd();

          return ret;
        }
      };

      template< unsigned N >
      struct init_zero_storage_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute()
        {
          storage_type< double, N > ret{};

          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.chunks[i] = _mm256_setzero_pd();

          return ret;
        }
      };

      template< unsigned N >
      struct init_set1_storage_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute( double _scalar )
        {
          storage_type< double, N > ret{};

          auto setter = _mm256_set1_pd( _scalar );

          for ( unsigned i = 0; i < storage_type< double, N >::num_unmasked; ++i )
            ret.chunks[i] = setter;
          if ( storage_type< double, N >::mask )
          {
            auto masked
                = _mm256_blendv_pd( _mm256_setzero_pd(), setter, detail::make_mask( storage_type< double, N >::mask ) );
            ret.chunks[storage_type< double, N >::num_unmasked] = masked;
          }

          return ret;
        }
      };

      template< unsigned N >
      struct init_load_storage_op< vec, double, double, N >
      {
        using result_type = double;

        static auto
        execute( const double *_data )
        {
          storage_type< double, N > ret{};

          static constexpr auto chunk_size = storage_type< double, N >::chunk_size;
          for ( unsigned i = 0; i < storage_type< double, N >::num_unmasked; ++i )
            ret.chunks[i] = _mm256_load_pd( _data + i * chunk_size );
          if ( storage_type< double, N >::mask )
          {
            static constexpr auto i = storage_type< double, N >::num_unmasked;
            ret.chunks[i]
                = _mm256_maskload_pd( _data + i * chunk_size, detail::make_imask64( storage_type< double, N >::mask ) );
          }

          return ret;
        }
      };

      template< unsigned N >
      struct add_op< vec, double, double, N >
      {
        using result_type = double;
        static auto
        execute( const vec< double, N > &_lhs, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_add_pd( _lhs.d.chunks[i], _rhs.d.chunks[i] );
          return ret;
        }
      };

      template< unsigned N >
      struct sub_op< vec, double, double, N >
      {
        using result_type = double;
        static auto
        execute( const vec< double, N > &_lhs, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_sub_pd( _lhs.d.chunks[i], _rhs.d.chunks[i] );
          return ret;
        }
      };

      template< unsigned N >
      struct mul_op< vec, double, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_lhs, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_mul_pd( _lhs.d.chunks[i], _rhs.d.chunks[i] );
          return ret;
        }

        static auto
        execute( const vec< double, N > &_lhs, double _scalar )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_mul_pd( _lhs.d.chunks[i], _mm256_set1_pd( _scalar ) );
          return ret;
        }

        static auto
        execute( double _scalar, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_mul_pd( _mm256_set1_pd( _scalar ), _rhs.d.chunks[i] );
          return ret;
        }
      };

      template< unsigned N >
      struct div_op< vec, double, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_lhs, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_div_pd( _lhs.d.chunks[i], _rhs.d.chunks[i] );
          return ret;
        }

        static auto
        execute( const vec< double, N > &_lhs, double _scalar )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_div_pd( _lhs.d.chunks[i], _mm256_set1_pd( _scalar ) );
          return ret;
        }

        static auto
        execute( double _scalar, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_div_pd( _mm256_set1_pd( _scalar ), _rhs.d.chunks[i] );
          return ret;
        }
      };

      template< unsigned N >
      struct haddv_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_vec )
        {
          vec< result_type, N > ret;

          // Summ all chunks
          __m256d sum = _mm256_setzero_pd();
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            sum = _mm256_add_pd( sum, _vec.d.chunks[i] );

          // hadd each 128-bit lane
          sum = _mm256_hadd_pd( sum, sum );
          sum = _mm256_add_pd( sum, _mm256_permute2f128_pd( sum, sum, 0x1 ) );

          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = sum;

          // sum the lanes
          return ret;
        }
      };

      template< unsigned N >
      struct sqrt_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_vec )
        {
          vec< result_type, N > ret;

          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_sqrt_pd( _vec.d.chunks[i] );

          return ret;
        }
      };

      template< unsigned N >
      struct floor_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_vec )
        {
          vec< result_type, N > ret;

          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_floor_pd( _vec.d.chunks[i] );

          return ret;
        }
      };

      template< unsigned N >
      struct ceil_op< vec, double, N >
      {
        using result_type = double;

        static auto
        execute( const vec< double, N > &_vec )
        {
          vec< result_type, N > ret;

          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_ceil_pd( _vec.d.chunks[i] );

          return ret;
        }
      };

      template< unsigned N >
      struct equals_op< double, double, N >
      {
        using result_type = double;
        static auto
        execute( const vec< double, N > &_lhs, const vec< double, N > &_rhs )
        {
          vec< result_type, N > ret;
          for ( unsigned i = 0; i < storage_type< double, N >::num_chunks; ++i )
            ret.d.chunks[i] = _mm256_cmp_pd( _lhs.d.chunks[i], _rhs.d.chunks[i], _CMP_EQ_OQ );
          return ret;
        }
      };


      template< unsigned N >
      struct truthiness_op< double, N >
      {
        using result_type = bool;

        static auto
        execute( const vec< double, N > &_vec )
        {
          for ( unsigned i = 0; i < storage_type< double, N >::num_unmasked; ++i )
            if ( _mm256_testz_pd( _vec.d.chunks[i], _vec.d.chunks[i] ) ) return false;
          if ( storage_type< double, N >::mask )
          {
            auto v = _vec.d.chunks[storage_type< double, N >::num_unmasked];
            auto masked
                = _mm256_blendv_pd( detail::make_mask( 0xf ), v, detail::make_mask( storage_type< double, N >::mask ) );
            return !_mm256_testz_pd( masked, masked );
          }

          return true;
        }
      };
    } // namespace ops
  } // namespace m
} // namespace bvh

#endif // BVH_SIMD

#endif // INC_BVH_MATH_OPS_AVX2_VEC_OPS_HPP
