/*
 * @Author: ycdhq 
 * @Date: 2023-04-13 16:49:02 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-07-10 17:04:04
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>
#include <iostream>
#include <filesystem>
#include <assert.h>
#include <string.h>
#include "npy.h"
#include "utils.h"
#include "log.h"
#define INSTANCE_TIMEOUT 100000
#define IMG_HEIGHT 256
#define IMG_WIDTH 704
constexpr int K = 500;
int out_size_factor_ = 8;
float voxel_size_[2] = {0.1, 0.1};
float pc_range_[2] = {-51.2, -51.2};
float post_center_range_[6] = {-61.2, -61.2, -10.0, 61.2, 61.2, 10.0};
int post_max_size_ = 500;
float nms_thr_[6] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
float factor_[6] = {1.0,0.7,0.4,1.0,1.0,4.5};
std::vector<std::string> nms_type = {"rotate", "rotate", "rotate", "circle", "rotate", "rotate"};
struct RotatedBox {
  float x_ctr, y_ctr, w, h, a;
};
using namespace std;

void load_npy(std::string file, std::vector<unsigned long> &shape, std::vector<float> &data) {
    bool is_fortran;
    // load ndarray voxel as vector<float>
    npy::LoadArrayFromNumpy(file, shape, is_fortran, data);
}


int decode(const out_tensor& regtb,
                  const out_tensor& heitb,
                  const out_tensor& dimtb,
                  const out_tensor& rottb,
                  const out_tensor& heatmaptb, float score_threshold,
                  uint32_t& first_label, std::vector<BBox>& res) {
  std::vector<float> heatmap = heatmaptb.second;
  int heatmap_shape = heatmaptb.first;

  // float heatmap_sigmoid[heatmap_shape * 128 * 128];
  // sigmoid_on_tensor(heatmap.data(), heatmap_sigmoid, heatmap_shape * 128 * 128);


  std::vector<float> reg = regtb.second;
  int reg_shape = regtb.first;

  std::vector<float> hei = heitb.second;
  int hei_shape = heitb.first;

  std::vector<float> dim = dimtb.second;
  int dim_shape = dimtb.first;

  std::vector<float> rot = rottb.second;
  int rot_shape = rottb.first;

  int cat = heatmap_shape;

  auto w = 128;
  std::vector<float> top_scores;
  std::vector<uint32_t> top_inds;
  std::vector<uint32_t> top_clses;
  std::vector<uint32_t> top_ys;
  std::vector<uint32_t> top_xs;
  if (cat == 1) {
    std::tie(top_scores, top_inds) = topK(heatmap.data(), heatmap_shape * 128 * 128, K, score_threshold);
    top_clses = std::vector<uint32_t>(top_scores.size(), 0);
    top_ys = division_n(top_inds, w);
    top_xs = remainder_n(top_inds, w);
  } else if (cat == 2) {
    std::vector<float> topk_scores;
    std::vector<uint32_t> topk_inds;
    std::tie(topk_scores, topk_inds) =
        topK(heatmap.data(), 128 * 128, 0, K, score_threshold);

    auto cls_flag = topk_inds.size();
    auto temp = topK(heatmap.data(), 128 * 128, 1, K, score_threshold);

    topk_scores.insert(topk_scores.end(), temp.first.begin(), temp.first.end());
    topk_inds.insert(topk_inds.end(), temp.second.begin(), temp.second.end());
    std::vector<uint32_t> topk_ind;
    std::tie(top_scores, topk_ind) = topK(topk_scores, K, score_threshold);

    top_inds = gather_feat(topk_inds, topk_ind);     
    for (auto&& i : topk_ind) {
      if (i < cls_flag)
        top_clses.push_back(0);
      else
        top_clses.push_back(1);
    }
    top_ys = gather_feat(division_n(topk_inds, w), topk_ind);
    top_xs = gather_feat(remainder_n(topk_inds, w), topk_ind);
  } else {
    AERROR << "The number of cat is not supported to be " << cat << std::endl;
  }

  int res_num = 0;

  if (top_scores.size() > 0) {
    auto reg_trans = reshape(reg.data(), reg_shape);
    auto top_reg = gather_feat(reg_trans, top_inds, 2);

    auto top_hei = gather_feat(hei, top_inds);

    auto rot_trans = reshape(rot.data(), rot_shape);  
    auto top_rot = gather_feat(rot_trans, top_inds, 2);

    auto dim_trans = reshape(dim.data(), dim_shape);
    auto top_dim = gather_feat(dim_trans, top_inds, 3);

    for (auto i = 0u; i < top_scores.size(); i++) {
      if (top_hei[i] >= post_center_range_[2] &&
          top_hei[i] <= post_center_range_[5]) {
        BBox tmp_result;
        float xs = (top_xs[i] + top_reg[i * 2]) * out_size_factor_ * voxel_size_[0] + pc_range_[0];
        float ys = (top_ys[i] + top_reg[i * 2 + 1]) * out_size_factor_ * voxel_size_[1] + pc_range_[1];
        if (xs >= post_center_range_[0] && ys >= post_center_range_[1] &&
            xs <= post_center_range_[3] && ys <= post_center_range_[4]) {
          tmp_result.bbox[0] = xs;
          tmp_result.bbox[1] = ys;

          tmp_result.bbox[3] = top_dim[3 * i];//std::exp(top_dim[3 * i]);
          tmp_result.bbox[4] = top_dim[3 * i + 1];//std::exp(top_dim[3 * i + 1]);
          tmp_result.bbox[5] = top_dim[3 * i + 2];//std::exp(top_dim[3 * i + 2]);
          
          tmp_result.bbox[2] =
              top_hei[i];// - tmp_result.bbox[5] * 0.5f;
          tmp_result.bbox[6] =
              atan2(top_rot[2 * i], top_rot[2 * i + 1]);
          tmp_result.score = top_scores[i];//sigmoid(top_scores[i]);//sigmoid(top_scores[i]);//top_scores[i];
          tmp_result.label = top_clses[i] + first_label;
          res.push_back(tmp_result);
          res_num++;
 
        }
      }
    }
  }
  first_label += cat;
  return res_num;

}

void get_rotated_vertices(const RotatedBox& box, Point (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  // MODIFIED
  double theta = box.a;
  float cosTheta2 = (float)cos(theta) * 0.5f;
  float sinTheta2 = (float)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

float cross_2d(const Point& A, const Point& B) {
  return A.x * B.y - B.x * A.y;
}

float dot_2d(const Point& A, const Point& B) {
  return A.x * B.x + A.y * B.y;
}

int get_intersection_points(const Point (&pts1)[4], const Point (&pts2)[4],
                            Point (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0;  // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      float det = cross_2d(vec2[j], vec1[i]);
      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      float t1 = cross_2d(vec2[j], vec12) / det;
      float t2 = cross_2d(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d(AB, AB);
    auto ADdotAD = dot_2d(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d(AP, AB);
      auto APdotAD = -dot_2d(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d(AB, AB);
    auto ADdotAD = dot_2d(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d(AP, AB);
      auto APdotAD = -dot_2d(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

int convex_hull_graham(
    const Point (&p)[24],
    const int& num_in,
    Point (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  float dist[24];

  // CPU version
  std::sort(
      q + 1, q + num_in, [](const Point& A, const Point& B) -> bool {
        float temp = cross_2d(A, B);
        if (fabs(temp) < 1e-6) {
          return dot_2d(A, A) < dot_2d(B, B);
        } else {
          return temp > 0;
        }
      });
  // compute distance to origin after sort, since the points are now different.
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d(q[i], q[i]);
  }

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1) {
      auto q1 = q[i] - q[m - 2], q2 = q[m - 1] - q[m - 2];
      // cross_2d() uses FMA and therefore computes round(round(q1.x*q2.y) -
      // q2.x*q1.y) So it may not return 0 even when q1==q2. Therefore we
      // compare round(q1.x*q2.y) and round(q2.x*q1.y) directly. (round means
      // round to nearest floating point).
      if (q1.x * q2.y >= q2.x * q1.y)
        m--;
      else
        break;
    }
    // Using double also helps, but float can solve the issue for now.
    // while (m > 1 && cross_2d<T, double>(q[i] - q[m - 2], q[m - 1] - q[m - 2])
    // >= 0) {
    //     m--;
    // }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }
  return m;
}


float polygon_area(const Point (&q)[24], int m) {
  if (m <= 2) {
    return 0;
  }

  float area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

float rotated_boxes_intersection(const RotatedBox& box1,
                                                const RotatedBox& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point intersectPts[24], orderedPts[24];

  Point pts1[4];
  Point pts2[4];
  get_rotated_vertices(box1, pts1);
  get_rotated_vertices(box2, pts2);

  int num = get_intersection_points(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham(intersectPts, num, orderedPts, true);
  return polygon_area(orderedPts, num_convex);
}


float single_box_iou_rotated(const float* box1_raw, const float* box2_raw) {
  // shift center to the middle point to achieve higher precision in result
  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  RotatedBox box1, box2;

  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];
  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

  const float area1 = box1.w * box1.h;
  const float area2 = box2.w * box2.h;
  if (area1 < 1e-14 || area2 < 1e-14) {
    return 0.f;
  }

  const float intersection = rotated_boxes_intersection(box1, box2);
  float baseS = (area1 + area2 - intersection);
  const float iou = intersection / baseS;
  return iou;
}

std::vector<int> nms_rotated(const float bboxes_xyxyr[][5], float* bboxes_scores, float iou_threshold, int decode_res_num) {

  std::vector<int> suppressed(decode_res_num, 0);
  std::vector<int> keep(decode_res_num, -1);  

  int num_to_keep = 0;
  for (int64_t _i = 0; _i < decode_res_num; _i++) {
    //auto i = order[_i];
    auto i = _i;
    if (suppressed[i] == 1) {
      continue;
    }

    keep[num_to_keep++] = i;

    for (int64_t _j = _i + 1; _j < decode_res_num; _j++) {
      //auto j = order[_j];
      auto j = _j;
      if (suppressed[j] == 1) {
        continue;
      }

      auto ovr = single_box_iou_rotated(
          bboxes_xyxyr[i], bboxes_xyxyr[j]);
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  return keep;
}


int main(int argc, char* argv[])
{
    std::vector<std::string> task_list = {"reg", "hei", "dim", "rot", "heatmap"};
    std::vector<std::vector<out_tensor>> output;
    uint32_t first_label = 0;
      
    
    std::vector<BBox> result;

    for (int i = 0; i < 6; i++) {
        std::vector<out_tensor> output_tensors;
        std::vector<BBox> res;
        for (auto & file : task_list) {
            std::string npy_file = "../decode_data_2/" + std::to_string(i) + "_" + file + ".npy";
            std::vector<unsigned long> shape;
            std::vector<float> data;
            load_npy(npy_file, shape, data);
            AINFO <<  shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3];      
            int cat = shape[1];
            output_tensors.push_back(std::make_pair(cat, data));
        }
        int decode_res_num = decode(output_tensors[0],
                                    output_tensors[1],
                                    output_tensors[2],
                                    output_tensors[3],
                                    output_tensors[4],
                                    0.25, first_label, res);
        float bboxes_xyxyr[decode_res_num][5];
        float bboxes_scores[decode_res_num];

        if (nms_type[i] == "rotate") {
          for (auto n = 0; n < decode_res_num; ++n) {
            bboxes_scores[n] = res[n].score;
            xywhr2xyxyr(res[n].bbox, bboxes_xyxyr[n], factor_[i]);
          }
          std::vector<int> keep = nms_rotated(bboxes_xyxyr, bboxes_scores, nms_thr_[i], decode_res_num);
          for (auto& k : keep) {
            if (k != -1)
              result.push_back(res[k]);
          }
        }
    }
    AINFO << result.size();
    return 0;
}
