#include <optimizer.h>

Optimizer::Optimizer():
    source_cloud_(new pcl::PointCloud<Point_T>()),
    target_cloud_(new pcl::PointCloud<Point_T>())
    {};

void Optimizer::optimize(){
    if(!source_cloud_ || source_cloud_->points.empty()) return;
    if(!target_cloud_ || target_cloud_->points.empty()) return;

    Matrix6f ATPA = Matrix6f::zeros();
    Vector6f ATPb = Vector6f::all(0.);

    float f_before = 0;

    for(auto& corr : correspondence_set_){
        sumPt2PtLLS(
            source_cloud_->points[corr.in_idx],
            target_cloud_->points[corr.ref_idx],
            f_before,
            &ATPA,
            &ATPb
        );
    }

    symmetrizeMatix(ATPA);

    Vector6f unknown_x;
    cv::solve(ATPA,ATPb, unknown_x);

    float step_size = 1.0;

    // wolfe_condition
    float pkt_grad = (unknown_x.t() * ATPb)(0);

    while (step_size > 0.01){
        float f_after = 0.0;

        Vector6f updated_x = unknown_x * step_size;
    
        pcl::PointCloud<Point_T>::Ptr temp_cloud(new pcl::PointCloud<Point_T>());
        updateCloud(source_cloud_, temp_cloud, updated_x);

        // calculate f_after
        for (auto& corr : correspondence_set_){
            sumPt2PtLLS(
                temp_cloud->points[corr.in_idx],
                target_cloud_->points[corr.ref_idx],
                f_after
            );
        }

        if(f_after <= f_before + 0.0001 * step_size * pkt_grad) break;

        step_size /= 2;
    }

    unknown_x *= step_size;
    result_transform_ = Eigen::Affine3f::Identity();
    vectorToAffine(unknown_x, result_transform_);
}

void Optimizer::setCorrespondence(std::vector<Correspondence> corr_vec){
    correspondence_set_ = corr_vec;
}

void Optimizer::symmetrizeMatix(Matrix6f& mat){
    mat(1,0) = mat(0,1);
    mat(2,0) = mat(0,2);
    mat(2,1) = mat(1,2);
    mat(3,0) = mat(0,3);
    mat(3,1) = mat(1,3);
    mat(3,2) = mat(2,3);
    mat(4,0) = mat(0,4);
    mat(4,1) = mat(1,4);
    mat(4,2) = mat(2,4);
    mat(4,3) = mat(3,4);
    mat(5,0) = mat(0,5);
    mat(5,1) = mat(1,5);
    mat(5,2) = mat(2,5);
    mat(5,3) = mat(3,5);
    mat(5,4) = mat(4,5);
}

void Optimizer::sumPt2PtLLS(Point_T in_point, Point_T ref_point, float& f, Matrix6f* ATPA, Vector6f* ATPb){
    float px = in_point.x;
    float py = in_point.y;
    float pz = in_point.z;
    float qx = ref_point.x;
    float qy = ref_point.y;
    float qz = ref_point.z;

    float dx = px - qx;
    float dy = py - qy;
    float dz = pz - qz;

    if(ATPA){
        (*ATPA)(0,0) += 1.0;
        (*ATPA)(0,1) += 0;
        (*ATPA)(0,2) += 0;
        (*ATPA)(0,3) += 0;
        (*ATPA)(0,4) += pz;
        (*ATPA)(0,5) += (-py);
        (*ATPA)(1,1) += 1.0;
        (*ATPA)(1,2) += 0;
        (*ATPA)(1,3) += (-pz);
        (*ATPA)(1,4) += 0;
        (*ATPA)(1,5) += px;
        (*ATPA)(2,2) += 1.0;
        (*ATPA)(2,3) += py;
        (*ATPA)(2,4) += (-px);
        (*ATPA)(2,5) += 0;
        (*ATPA)(3,3) += pz * pz + py * py;
        (*ATPA)(3,4) += (-px * py);
        (*ATPA)(3,5) += (-px * pz);
        (*ATPA)(4,4) += pz * pz + px * px;
        (*ATPA)(4,5) += (-py * pz);
        (*ATPA)(5,5) += py * py + px * px;
    }

    if(ATPb){
        (*ATPb)(0) += (-dx);
        (*ATPb)(1) += (-dy);
        (*ATPb)(2) += (-dz);
        (*ATPb)(3) += pz * dy - py * dz;
        (*ATPb)(4) += px * dz - pz * dx;
        (*ATPb)(5) += py * dx - px * dy;
    }

    f += dx * dx + dy * dy + dz * dz;
}

void Optimizer::updateCloud(pcl::PointCloud<Point_T>::Ptr in_cloud, pcl::PointCloud<Point_T>::Ptr updated_cloud, const Vector6f updated_x){
    Eigen::Affine3f update_transform = Eigen::Affine3f::Identity();
    vectorToAffine(updated_x, update_transform);
    pcl::transformPointCloud(*source_cloud_, *updated_cloud, update_transform);
}

void Optimizer::vectorToAffine(const Vector6f& in_vec, Eigen::Affine3f& out_affine){
    Eigen::AngleAxisf rotation_x(in_vec(3), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf rotation_y(in_vec(4), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rotation_z(in_vec(5), Eigen::Vector3f::UnitZ());

    out_affine.rotate(rotation_z * rotation_y * rotation_x);
    out_affine.translation() << in_vec(0), in_vec(1), in_vec(2);
}