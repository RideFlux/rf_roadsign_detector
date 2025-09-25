
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <opencv2/core.hpp>

using Matrix6f = cv::Matx<float,6,6>;
using Vector6f = cv::Vec<float,6>;

typedef pcl::PointXYZ Point_T;

class Optimizer
{
public: 
    struct Correspondence{
        int in_idx;
        int ref_idx;
        Correspondence(int in_idx_, int ref_idx_) : in_idx(in_idx_), ref_idx(ref_idx_){}
    };
    Optimizer();

    void optimize();
    void setSourceCloud(pcl::PointCloud<Point_T>::Ptr in_cloud){source_cloud_ = in_cloud;}
    void setTargetCloud(pcl::PointCloud<Point_T>::Ptr in_cloud){target_cloud_ = in_cloud;}
    void setCorrespondence(std::vector<Correspondence> corr_vec);
    void clear(){correspondence_set_.clear();}

    inline bool isSourceCloudEmpty() const {return source_cloud_->points.empty();}
    inline bool isTargetCloudEmpty() const {return target_cloud_->points.empty();}
    auto& getResult() const{return result_transform_;}

private: 
    void symmetrizeMatix(Matrix6f& mat);
    void sumPt2PtLLS(Point_T in_point, Point_T ref_point, float& f, Matrix6f* ATPA = nullptr, Vector6f* ATPb = nullptr);
    void updateCloud(pcl::PointCloud<Point_T>::Ptr in_cloud, pcl::PointCloud<Point_T>::Ptr updated_cloud, const Vector6f updated_x);
    void vectorToAffine(const Vector6f& in_vec, Eigen::Affine3f& out_affine);

private:
    std::vector<Correspondence> correspondence_set_;
    pcl::PointCloud<Point_T>::Ptr source_cloud_;
    pcl::PointCloud<Point_T>::Ptr target_cloud_;

    Eigen::Affine3f result_transform_;
};
